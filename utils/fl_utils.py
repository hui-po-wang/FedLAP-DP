import copy
import numpy as np
import time
from tqdm import tqdm
from random import shuffle

# import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

from utils.augmentation import DiffAugment
from utils.utils import TensorDataset, compute_model_diff
from utils.misc import get_network
from utils.ops import match_loss, epoch

from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus.privacy_engine import PrivacyEngine
from rdp_accountant import compute_sigma

class VirtualWorker():
    def __init__(self, wid, state_server, args):
        self.wid = wid
        self.state_server = state_server
        self.args = args
         
        self.state = None
        self.dset = None
        self.loader = None
        self.opt = None
        self.opt_img = None

        # used for dataset condensation 
        self.reinit_imgs = True
        self.indices_class = [[] for _ in range(state_server['num_classes'])]
        self.image_syn = None
        self.label_syn = None
        # record unique labels in a set
        self.label_set = set()

    def get_images(self, c, n): # get random n images from class c
        assert ('fedlap' in self.args.optimization or 'GM' in self.args.optimization) and isinstance(self.label_set, list)
        assert len(self.indices_class[c]) > 0, 'this client does not have class {c}.'
        idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
        return self.images_all[idx_shuffle], self.labels_all[idx_shuffle]

    def get_num_classes(self):
        return len(self.label_set)
        
    def set_dset(self, dset):
        self.dset = dset
        self.images_all = [torch.unsqueeze(dset[i][0], dim=0) for i in range(len(dset))]
        self.labels_all = [dset[i][1] for i in range(len(dset))]
        self.images_all = torch.cat(self.images_all, dim=0)
        self.labels_all = torch.tensor(self.labels_all, dtype=torch.long)
        # for federated learning, clients may only have access to a subset of labels
        for ind, (img, lbl) in enumerate(self.dset):
            self.indices_class[lbl].append(ind)
            # contains the classes on a client, used for enumeration
            self.label_set.add(lbl.item())
        self.label_set = list(self.label_set)
        self.inf_loader = self.inf_train_gen()

    def set_loader(self, loader):
        self.loader = loader

    def set_opt(self, opt):
        self.opt = opt

    def set_opt_img(self, opt_img):
        self.opt_img = opt_img

    def inf_train_gen(self):
        while True:
            for images, targets in self.loader:
                yield (images, targets)

    def init_state(self, state):
        self.state = state
        self.state.apply(_zero_weights)
        self.state.requires_grad = False
        self.state.to('cpu')

def _zero_weights(m):
    for p in m.parameters():
        torch.nn.init.constant_(p, 0)

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_model(src, dst, args):
    for s, d in zip( src.parameters() , dst.parameters() ):
        d.data = s.data.detach().clone()

def prepare_workers(args, trainset, state_server):
    # Create number of virtual workers that will act as clients
    workers = {}
    for i in range(args.n_client):
        workers[i] = VirtualWorker(i, state_server, args)
        print(len(trainset[i]), i)
        sampling_rate = args.batch_size / len(trainset[i])
        workers[i].sampling_rate = sampling_rate
        
        uniform_sampler = UniformWithReplacementSampler(num_samples=len(trainset[i]), sample_rate=sampling_rate)
        trainloader = torch.utils.data.DataLoader(trainset[i], batch_sampler=uniform_sampler, pin_memory=True)

        workers[i].set_dset(trainset[i])
        workers[i].set_loader(trainloader)

    return workers
        
def compute_model_distance(m1, m2):
    # This function returns the Euclidean distance between two models. A float on cpu will be returned.
    with torch.no_grad():
        stacked_m1 = torch.stack([p.detach().norm(p=2) for p in m1.parameters()])
        stacked_m2 = torch.stack([p.detach().norm(p=2) for p in m2.parameters()])
        dis = torch.sqrt(torch.sum((stacked_m1 - stacked_m2) ** 2)).item()
    return dis

def test(args, model, device, test_loader, result, verbose=True):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()

    if verbose:
        print('Test set: Accuracy: {}/{} ({:.2f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    result.append( 100. * correct / len(test_loader.dataset) )

    model.train()

def fedm(args, state_server, current_worker, model, metric, reset_methods=None, reinit_image=False, verbose=False, target_grad=None):
    num_classes = state_server['num_classes']
    channel = state_server['channel']
    im_size = state_server['im_size']

    # Note that due to the partition, some clients might not have all claases
    num_classes_client = current_worker.get_num_classes()
    # initialize synthetic images with random noise
    if reinit_image or current_worker.image_syn is None:
        current_worker.image_syn = torch.randn(size=(num_classes_client*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        if args.init == 'real':
            for c_ind, c in enumerate(current_worker.label_set):
                current_worker.image_syn.data[c_ind*args.ipc:(c_ind+1)*args.ipc] = current_worker.get_images(c, args.ipc)[0].detach().data.view(current_worker.image_syn.data[c_ind*args.ipc:(c_ind+1)*args.ipc].size())
        current_worker.label_syn = np.array([np.ones(args.ipc)*i for i in current_worker.label_set])
        current_worker.label_syn = torch.tensor(current_worker.label_syn, dtype=torch.long, requires_grad=False, device=args.device).view(-1)

    image_syn = current_worker.image_syn
    label_syn = current_worker.label_syn
    print(image_syn.size())
    optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)

    if args.enable_privacy: 
        iters = args.iteration * args.outer_loop * args.batch_loop
        if args.noise_multiplier is None:
            noise_multiplier = compute_sigma(args.target_epsilon, current_worker.sampling_rate, iters, args.target_delta)
        else:
            noise_multiplier = args.noise_multiplier

    loss_avg = [0 for _ in range(args.boost_loop)]
    loss_reg_avg = [0 for _ in range(args.boost_loop)]
    for it in tqdm(range(args.iteration)):  ## re-initialize the model for each iter
        if reset_methods == 'current_weight':
            net = copy.deepcopy(model)
        elif reset_methods == 'random':
            net = get_network(args.arch, channel, num_classes, im_size).to(args.device)  # get a random model
        else:
            raise NotImplementedError(f'Unknown methods for resetting models when syntheisizing images: {reset_methods}')

        net_shadow = copy.deepcopy(net)  # Used for obtain DP real gradient (shadow is necessary as otherwise the hooks will cause problems)

        net_parameters = list(net.parameters())
        net_shadow_parameters = list(net_shadow.parameters())  

        optimizer_net = optim.SGD(params=net.parameters(), lr=args.lr_net)  # optimizer for update model
        optimizer_net_grad = optim.SGD(params=net_shadow.parameters(), lr=args.lr_net) # optimizer for obtaining DP real gradient

        args.dc_aug_param = None  # Mute the DC augmentation when training synthetic data.

        ### Initialize for DP
        if args.enable_privacy:
            ### Initialize privacy engine
            privacy_engine = PrivacyEngine(net_shadow, sample_rate=current_worker.sampling_rate, alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)), noise_multiplier=noise_multiplier, max_grad_norm=args.max_norm)
            privacy_engine.attach(optimizer_net_grad)

        for ol in range(args.outer_loop): # used to avoid infinite-loop cases
            current_diff = compute_model_diff(net, model)
            if current_diff >= args.eps_ball: #max_approx_distance
                print(f'exceed the defined region {current_diff}')
                break

            ### Optimize synthetic data
            real_grad_list = []
            for _ in range(args.batch_loop):  # sample multiple batches of real data and obtain gradients given the same model parameter (~target at local behavior matching)
                img_real, lab_real = next(current_worker.loader.__iter__())
                img_real = img_real.to(args.device)
                lab_real = lab_real.to(args.device)

                sample_index = list(range(image_syn.size(0)))
                shuffle(sample_index)
                img_syn = image_syn[sample_index[:args.server_batch_size]]
                lab_syn = label_syn[sample_index[:args.server_batch_size]]

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param).detach()
                    img_syn = DiffAugment(image_syn[sample_index[:args.server_batch_size]], args.dsa_strategy, seed=seed, param=args.dsa_param)

                ## Compute real_gradient
                net.train()
                net_shadow.train()
                if args.enable_privacy:
                    net_shadow.load_state_dict(net.state_dict())  # synchronize the current model parameter (net -> net_shadow)
                    net_shadow.zero_grad()

                    output_real = net_shadow(img_real)
                    loss_real = metric(output_real, lab_real)
                    loss_real.backward()
                    optimizer_net_grad.step()  # this step compute the DP noisy gradient on net_shadow
                    
                    gw_real = list((p.grad.detach().clone() for p in net_shadow_parameters))
                else:
                    if args.dis_metric == 'fm':
                        gw_real = []
                        def save_features(self, input, output):
                            gw_real.append(output.mean(dim=0))
                        for n, m in net.named_modules():
                            if isinstance(m, nn.ReLU()):
                                m.register_forward_hook(hook=save_features)
                        output_real = net(img_real)
                    else:
                        net.zero_grad()
                        output_real = net(img_real)
                        loss_real = metric(output_real, lab_real)
                        gw_real = torch.autograd.grad(loss_real, net_parameters)
                        gw_real = list((_.detach().clone() for _ in gw_real))

                        if verbose:
                            grad_list = list((_.detach().clone().norm().item() for _ in gw_real))
                            real_grad_list.append(np.mean(grad_list))
                net.eval()
                net_shadow.eval()

                ## Compute fake_gradient and matching loss
                for i_boost in range(args.boost_loop):
                    net.zero_grad()
                    output_syn = net(img_syn)
                    loss_syn = metric(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    loss = match_loss(gw_syn, gw_real, args)
                    # record conventional losses
                    loss_avg[i_boost] += loss.item()
                    if args.reg_type is not None:
                        reg_loss = args.reg_weight * match_loss(gw_syn, gw_real, args, dis_metric=args.reg_type)
                        # record regualrization losses, if any
                        loss_reg_avg[i_boost] += reg_loss.item()
                        loss += reg_loss
                        
                    ## Update image
                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()

                    if args.dsa:
                        img_syn = DiffAugment(image_syn[sample_index[:args.server_batch_size]], args.dsa_strategy, seed=seed, param=args.dsa_param)

            if ol == args.outer_loop - 1:
                break

            ### Update network (#inner_loop epochs on the current synthetic set)
            image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
            dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
            trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.server_batch_size, shuffle=True, num_workers=0)
            
            for il in range(args.inner_loop):
                epoch('train', trainloader, net, optimizer_net, metric, args, aug=True if args.dsa else False)

        if verbose:            
            return np.array(loss_avg) / (args.outer_loop * args.batch_loop * args.iteration), np.array(loss_reg_avg) / (args.outer_loop * args.batch_loop * args.iteration), real_grad_list
        else:
            return np.array(loss_avg) / (args.outer_loop * args.batch_loop * args.iteration), np.array(loss_reg_avg) / (args.outer_loop * args.batch_loop * args.iteration)
        
def gen_syn_image(args, state_server, current_worker, model, metric, reset_methods=None, reinit_image=False, verbose=False, target_grad=None):
    num_classes = state_server['num_classes']
    channel = state_server['channel']
    im_size = state_server['im_size']

    # Note that due to the partition, some clients might not have all claases
    num_classes_client = current_worker.get_num_classes()
    # initialize synthetic images with random noise
    if reinit_image or current_worker.image_syn is None:
        current_worker.image_syn = torch.randn(size=(num_classes_client*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        if args.init == 'real':
            for c_ind, c in enumerate(current_worker.label_set):
                current_worker.image_syn.data[c_ind*args.ipc:(c_ind+1)*args.ipc] = current_worker.get_images(c, args.ipc)[0].detach().data.view(current_worker.image_syn.data[c_ind*args.ipc:(c_ind+1)*args.ipc].size())
        current_worker.label_syn = np.array([np.ones(args.ipc)*i for i in current_worker.label_set])
        current_worker.label_syn = torch.tensor(current_worker.label_syn, dtype=torch.long, requires_grad=False, device=args.device).view(-1)

    image_syn = current_worker.image_syn
    label_syn = current_worker.label_syn
    print(image_syn.size())
    optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)

    if args.enable_privacy: 
        iters = args.iteration * args.outer_loop * args.batch_loop
        if args.noise_multiplier is None:
            noise_multiplier = compute_sigma(args.target_epsilon, current_worker.sampling_rate, iters, args.target_delta)
        else:
            noise_multiplier = args.noise_multiplier

    loss_avg = [0 for _ in range(args.boost_loop)]
    loss_reg_avg = [0 for _ in range(args.boost_loop)]
    for it in tqdm(range(args.iteration)):  ## re-initialize the model for each iter
        if reset_methods == 'current_weight':
            net = copy.deepcopy(model)
        elif reset_methods == 'random':
            net = get_network(args.arch, channel, num_classes, im_size).to(args.device)  # get a random model
        else:
            raise NotImplementedError(f'Unknown methods for resetting models when syntheisizing images: {reset_methods}')

        net_shadow = copy.deepcopy(net)  # Used for obtain DP real gradient (shadow is necessary as otherwise the hooks will cause problems)

        net_parameters = list(net.parameters())
        net_shadow_parameters = list(net_shadow.parameters())  

        optimizer_net = optim.SGD(params=net.parameters(), lr=args.lr_net)  # optimizer for update model
        optimizer_net_grad = optim.SGD(params=net_shadow.parameters(), lr=args.lr_net) # optimizer for obtaining DP real gradient

        args.dc_aug_param = None  # Mute the DC augmentation when training synthetic data.

        ### Initialize for DP
        if args.enable_privacy:
            ### Initialize privacy engine
            privacy_engine = PrivacyEngine(net_shadow, sample_rate=current_worker.sampling_rate, alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)), noise_multiplier=noise_multiplier, max_grad_norm=args.max_norm)
            privacy_engine.attach(optimizer_net_grad)

        for ol in range(args.outer_loop): # used to avoid infinite-loop cases
            current_diff = compute_model_diff(net, model)
            if current_diff >= args.eps_ball: #max_approx_distance
                break

            ### Optimize synthetic data
            real_grad_list = []
            for _ in range(args.batch_loop):  # sample multiple batches of real data and obtain gradients given the same model parameter (~target at local behavior matching)
                img_real, lab_real = next(current_worker.loader.__iter__())
                img_real = img_real.to(args.device)
                lab_real = lab_real.to(args.device)

                sample_index = list(range(image_syn.size(0)))
                shuffle(sample_index)
                img_syn = image_syn[sample_index[:args.server_batch_size]]
                lab_syn = label_syn[sample_index[:args.server_batch_size]]

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param).detach()
                    img_syn = DiffAugment(image_syn[sample_index[:args.server_batch_size]], args.dsa_strategy, seed=seed, param=args.dsa_param)

                ## Compute real_gradient
                net.train()
                net_shadow.train()
                if args.enable_privacy:
                    net_shadow.load_state_dict(net.state_dict())  # synchronize the current model parameter (net -> net_shadow)
                    net_shadow.zero_grad()

                    output_real = net_shadow(img_real)
                    loss_real = metric(output_real, lab_real)
                    loss_real.backward()
                    optimizer_net_grad.step()  # this step compute the DP noisy gradient on net_shadow
                    
                    gw_real = list((p.grad.detach().clone() for p in net_shadow_parameters))
                else:
                    if args.dis_metric == 'fm':
                        gw_real = []
                        def save_features(self, input, output):
                            gw_real.append(output.mean(dim=0))
                        for n, m in net.named_modules():
                            if isinstance(m, nn.ReLU()):
                                m.register_forward_hook(hook=save_features)
                        output_real = net(img_real)
                    else:
                        net.zero_grad()
                        output_real = net(img_real)
                        loss_real = metric(output_real, lab_real)
                        gw_real = torch.autograd.grad(loss_real, net_parameters)
                        gw_real = list((_.detach().clone() for _ in gw_real))

                        if verbose:
                            grad_list = list((_.detach().clone().norm().item() for _ in gw_real))
                            real_grad_list.append(np.mean(grad_list))
                net.eval()
                net_shadow.eval()

                ## Compute fake_gradient and matching loss
                for i_boost in range(args.boost_loop):
                    net.zero_grad()
                    output_syn = net(img_syn)
                    loss_syn = metric(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    loss = match_loss(gw_syn, gw_real, args)
                    # record conventional losses
                    loss_avg[i_boost] += loss.item()
                    if args.reg_type is not None:
                        reg_loss = args.reg_weight * match_loss(gw_syn, gw_real, args, dis_metric=args.reg_type)
                        # record regualrization losses, if any
                        loss_reg_avg[i_boost] += reg_loss.item()
                        loss += reg_loss
                        
                    ## Update image
                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()

                    if args.dsa:
                        img_syn = DiffAugment(image_syn[sample_index[:args.server_batch_size]], args.dsa_strategy, seed=seed, param=args.dsa_param)

            if ol == args.outer_loop - 1:
                break

            ### Update network (#inner_loop epochs on the current synthetic set)
            image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
            dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
            trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.server_batch_size, shuffle=True, num_workers=0)
            
            for il in range(args.inner_loop):
                epoch('train', trainloader, net, optimizer_net, metric, args, aug=True if args.dsa else False)

    if verbose:            
        return np.array(loss_avg) / (args.outer_loop * args.batch_loop * args.iteration), np.array(loss_reg_avg) / (args.outer_loop * args.batch_loop * args.iteration), real_grad_list
    else:
        return np.array(loss_avg) / (args.outer_loop * args.batch_loop * args.iteration), np.array(loss_reg_avg) / (args.outer_loop * args.batch_loop * args.iteration)

def update_model_global_optim(global_optim, model, buffer, test_loader, device, metric, epoch, args):
    model.train()
    if 'fedlap' in args.optimization:
        img_syn = torch.cat([p[0] for p in buffer['dsc_images']], 0)
        lbl_syn = torch.cat([p[1] for p in buffer['dsc_images']], 0)
        print(f'synthetic dataset size: {img_syn.size()}, and {lbl_syn.size()}')

        image_syn_train, label_syn_train = copy.deepcopy(img_syn.detach()), copy.deepcopy(lbl_syn.detach())  # avoid any unaware modification
        dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
        trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.server_batch_size, shuffle=True, num_workers=0)
        
        target_eps = None
        if args.truncate:
            if args.truncate_voting == 'max':
                target_eps = np.max(buffer['valid_region'])
            elif args.truncate_voting == 'min':
                target_eps = np.min(buffer['valid_region'])
            elif args.truncate_voting == 'avg':
                target_eps = np.average(buffer['valid_region'])
            elif args.truncate_voting == 'median':
                target_eps = np.median(buffer['valid_region'])
            elif args.truncate_voting == 'constant':
                target_eps = args.eps_ball
            else:
                raise NotImplementedError()
            net_source = copy.deepcopy(model)

        for it in tqdm(range(args.dsc_server_iter)):
            buffer['escape_epoch'] = it
            for id_batch, (img, lbl) in enumerate(trainloader):
                img, lbl = img.to(args.device), lbl.to(args.device)

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img = DiffAugment(img, args.dsa_strategy, seed=seed, param=args.dsa_param)
                out = model(img)
                loss = metric(out, lbl)

                global_optim.zero_grad()
                loss.backward()
                global_optim.step()

            if target_eps is not None:
                eps = compute_model_diff(model, net_source)
                if eps >= target_eps:
                    break
    elif args.optimization in ['fedavg', 'fedprox', 'fedadam', 'fedsgd']:
        ### FedAdam here
        global_optim.zero_grad()
        #import pdb; pdb.set_trace()
        for k, p in model.named_parameters(): 
            weight = 600 * len(buffer['gradient_data'])
            grad_out = 0
            n_nan = 0
            
            #print(k)
            for i in range(len(buffer['gradient_data'])):
                # TODO: check whether the name of the current grad exists in the buffer
                if not k in buffer['gradient_data'][i].keys():
                    break
                data = buffer['gradient_data'][i][k].cuda()
                grad_out += - data * 600 / weight

            if args.n_update_client > n_nan:
                if p.grad is None:
                    p.grad = -grad_out.cuda()
                else:
                    p.grad.add_( -grad_out.cuda() )
            
        global_optim.step()
    else:
        raise NotImplementedError()

def compute_client_gradients(model, model_new, buffer, args):
    gradient_data = {}
    gradient_rec1 = {}
    gradient_rec2 = {}
    gradient_rec3 = {}
        
    for m1, m2 in zip( model.named_parameters() , model_new.named_parameters() ):
        assert m1[0] == m2[0]
        assert m1[1].shape == m2[1].shape
        
        tmp = m1[1] - m2[1]
        gradient_data[m1[0]] = tmp.detach()
    
    buffer['gradient_data'].append(gradient_data)
    buffer['gradient_rec1'].append(gradient_rec1)
    buffer['gradient_rec2'].append(gradient_rec2)
    buffer['gradient_rec3'].append(gradient_rec3)

def loss_prox(model_global, model_local, device):
    loss = torch.tensor(0.0).to(device)
    for m1, m2 in zip( model_global.named_parameters() , model_local.named_parameters() ):
        assert m1[0] == m2[0]
        assert m1[1].shape == m2[1].shape
        
        loss += torch.norm(m1[1].detach() - m2[1]) ** 2
        
    return loss