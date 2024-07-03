# import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# import python library
import os
import random
import numpy as np
import argparse
import copy
import sys
import yaml
import time
from collections import defaultdict
from random import shuffle
from tqdm import tqdm

# import local library
from utils.fl_utils import update_model_global_optim, test, prepare_workers, gen_syn_image
from utils.utils import Parser, LearningScheduler, TensorDataset, FLLogger
from utils.misc import get_network, prepare_data, get_loops
from utils.augmentation import DiffAugment
from utils.analysis import measure_eps

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', '--cfg', default=None, type=str, required=True)
    parser.add_argument('-seed', '--seed', type=int, default=None)

    parser.add_argument('-data-path', '--data-path', default='./datasets', type=str)
    parser.add_argument('-download', '--download', action='store_true')

    parser.add_argument('-save_path', '--save_path', default='./saves', type=str)

    # if start-epoch != 1, load the pretrained model
    parser.add_argument('-start-epoch', '--start-epoch', default=1, type=int)
    parser.add_argument('-start-model', '--start-model', default=None, type=str)
    parser.add_argument('-start-log', '--start-log', default=None, type=str)

    parser.add_argument('-tag', '--tag', default=None, type=str)

    parser.add_argument('-verbose', '--verbose', action='store_true')
    parser.add_argument('-test_version', '--test_version', action='store_true')

    args = parser.parse_args()
    with open(args.cfg, 'r') as stream:
        settings = yaml.safe_load(stream)
    args = Parser(args, settings)
    args.name = os.path.basename(args.cfg).split('.')[0]
    if args.tag is not None:
        args.name = args.name + '-' + args.tag

    args.log_dir = os.path.join('runs/', args.arch, args.name)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # used for keeping all model weights and the configuration file, etc.
    args.train_dir = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    outer_loop, inner_loop = get_loops(args.ipc)  # obtain default setting (will be overwritten if specified)
    if args.outer_loop == -1:
        args.outer_loop = outer_loop
    if args.inner_loop == -1:
        args.inner_loop = inner_loop

    print(args)
    return args

def train(args, global_optim, model, state_server, metric,
    device, workers, current_epoch, buffer, lr_scheduler, test_loader, warmup=False):
    model.train()

    client_samples = list(range(args.n_client))

    # buffers for standard training
    buffer['gradient_data'] = []
    buffer['gradient_rec1'] = []
    buffer['gradient_rec2'] = []
    buffer['gradient_rec3'] = []
    # buffers for dataset distillation methods
    buffer['dsc_images'] = []
    buffer['valid_region'] = []
    buffer['escape_epoch'] = None

    shuffle(client_samples)

    for id_client in client_samples[:args.n_update_client]:
        current_worker = workers[id_client]

        args['tmp_buffer']['id_client'] = id_client
        loss_avg, loss_reg_avg = gen_syn_image(args, state_server, current_worker, model, metric, reset_methods=args.reset_method, reinit_image=args.reinit_image)
        args.clean_buf()
        args.logger.add_value(f'clients/{id_client}/matching_loss', loss_avg)
        args.logger.add_value(f'clients/{id_client}/matching_reg_loss', loss_reg_avg)

        print(f'client {id_client} loss: {loss_avg} Net LR: {args.lr_net}')
        if args.reg_type is not None:
            print(f'client {id_client} reg. loss: {loss_reg_avg}')

        # fectch synthetic images and labels
        image_syn = current_worker.image_syn
        label_syn = current_worker.label_syn
                
        # compute the valid region on every client, if needed
        # the truncation criterion is decided heuristically
        if args.truncate and args.truncate_voting != 'constant':
            image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
            dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
            trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.server_batch_size, shuffle=True, num_workers=0)

            net = copy.deepcopy(model)
            net_fixed = copy.deepcopy(model)
            net_fixed.eval() # never changed

            opt_net = optim.SGD(params=net.parameters(), lr=args.lr_net, momentum=args.momentum, weight_decay=args.weight_decay)

            stats_client = defaultdict(list)
            for i in tqdm(range(args.dsc_server_iter)):
                net.train()
                for img, lbl in trainloader:
                    img, lbl = img.to(args.device), lbl.to(args.device)

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img = DiffAugment(img, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        
                    pred = net(img)
                    loss = metric(pred, lbl)
                    
                    opt_net.zero_grad()
                    loss.backward()
                    opt_net.step()
                net.eval()
                with torch.no_grad():
                    (eps_syn, real_loss, syn_loss, real_acc,
                        syn_acc, grad_error, grad_cos) = measure_eps(args, net, net_fixed, metric, trainloader, current_worker.loader)
                
                args.logger.add_value(f'clients/{id_client}/eps_syn', eps_syn)
                args.logger.add_value(f'clients/{id_client}/real_loss', real_loss)
                args.logger.add_value(f'clients/{id_client}/syn_loss', syn_loss)
                args.logger.add_value(f'clients/{id_client}/real_acc', real_acc)
                args.logger.add_value(f'clients/{id_client}/syn_acc', syn_acc)
                args.logger.add_value(f'clients/{id_client}/grad_error', grad_error)
                args.logger.add_value(f'clients/{id_client}/grad_cos', grad_cos)
                
                stats_client['eps_syn'].append(eps_syn)
                stats_client['real_loss'].append(real_loss)
                stats_client['syn_loss'].append(syn_loss)
                stats_client['real_acc'].append(real_acc)
                stats_client['syn_acc'].append(syn_acc)
                stats_client['grad_error'].append(grad_error)
                stats_client['grad_cos'].append(grad_cos)
                
            returned_eps = None
            if args.truncate_crit == 'loss':
                index = np.argmin(stats_client['real_loss'])
                returned_eps = stats_client['eps_syn'][index]
            elif args.truncate_crit == 'grad_cos':
                index = np.argmin(np.array(stats_client['grad_cos']) - 1.0)
                returned_eps = stats_client['eps_syn'][index]
            else:
                raise NotImplementedError('')

            if returned_eps > args.eps_ball:
                returned_eps = args.eps_ball
            buffer['valid_region'].append(returned_eps)
            print(buffer['valid_region'])

        # saves the results in the buffer
        buffer['dsc_images'].append((image_syn.cpu(), label_syn.cpu()))

    update_model_global_optim(global_optim['optim'], model, buffer, test_loader, device, metric, current_epoch, args)

def create_server_opt(subnet_server, args):
    global_optim = {}
    if args.optimization == 'fedlap':
        global_optim['optim'] = optim.SGD(params=subnet_server.parameters(), lr=args.lr_net,
           momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        global_optim['optim'] = optim.SGD(params=subnet_server.parameters(), lr=args.lr_net)
    global_optim['optim_init'] = True
    return global_optim

def main(args):
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    state_server = {}
    # data
    # global information about the datasets is also recorded in state_server
    trainset, test_loader, split_in = prepare_data(args, use_cuda, state_server)
    # workers -- initialize workers according to the server state and args
    workers = prepare_workers(args, trainset, state_server)
    # Initialize the model
    model_server = get_network(args.arch, state_server['channel'], state_server['num_classes'], state_server['im_size']).to(args.device)

    n_param_model = 0
    for parameter in model_server.parameters(): n_param_model += parameter.nelement()
    print("# of model parameters: %d"%n_param_model)

    if args.start_epoch != 1 and args.start_model is not None:
        model_load_tmp = torch.load(args.start_model)
        model_server.load_state_dict(model_load_tmp["state_dict"] , strict=False)

        cur_pos = 0
        for i in range(args.n_client):
            next_pos = cur_pos + args.ipc * workers[i].get_num_classes()
            workers[i].image_syn = torch.tensor(model_load_tmp['img_syn'][cur_pos:next_pos], dtype=torch.float, requires_grad=True, device=args.device)
            workers[i].label_syn = torch.tensor(model_load_tmp['lbl_syn'][cur_pos:next_pos], dtype=torch.long, requires_grad=False, device=args.device)
            cur_pos = next_pos
        tmp_result = []

        test(args, model_server, device, test_loader, tmp_result)
        print(f'model loaded: {args.start_model} with acc: {tmp_result[-1]}')

    metric = nn.CrossEntropyLoss()
    args.logger = FLLogger(args, model=model_server)
    if args.start_log is not None:
        result_load_tmp = torch.load(args.start_log)['result']
        args.logger.load(result_load_tmp, epochs=args.start_epoch)

    print(model_server)

    global_optim = create_server_opt(model_server, args)

    lr_scheduler = LearningScheduler(args)

    global_optim = create_server_opt(model_server, args)
    lr_scheduler.set_opt(global_optim['optim'])
    init_lr = args.lr_net

    # log
    writer = SummaryWriter(args.log_dir)

    result = []
    accu_cost = 0

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        sys.stdout.flush()

        # record communication cost
        if 'fedlap' in args.optimization:
            max_num_class = -1e9
            for k, v in workers.items():
                max_num_class = max(max_num_class, len(v.label_set))
            cur_cost = args.ipc * max_num_class * state_server['channel'] \
                * state_server['im_size'][0] * state_server['im_size'][1]
        else:
            cur_cost = 0
            for parameter in model_server.parameters(): cur_cost += parameter.nelement()
        
        # megabytes
        accu_cost += args.n_update_client * (cur_cost*4/1000/1000)

        buffer = {}

        if args['client_settings']['type']  == 'cosine_decay':
            args.lr_net = max(init_lr * (1 + np.cos(np.pi * (epoch-1) / (args.epochs-1) ) ) / 2 , 1e-6)
        elif args['client_settings']['type']  == 'constant':
            pass
        elif args['client_settings']['type'] == 'multistep':
            args.lr_net = init_lr * (0.2 ** (epoch // 20))
        elif args['client_settings']['type'] == 'milestone':
            if epoch in args['client_settings']['milestones']:
                args.lr_net *= args['client_settings']['gamma']

        if 'fedlap' in args.optimization:
            # if use fedlap methods, reinitialize the global optimizer every round
            global_optim = create_server_opt(model_server, args)

        train(args, global_optim, model_server, state_server, metric, device, workers, epoch, buffer, lr_scheduler, test_loader)
        if epoch % args.test_interval == 0:
            start_time = time.time()
            test(args, model_server, device, test_loader, result)
            print("--- %s seconds for test---" % (time.time() - start_time))
            writer.add_scalar('Metric/acc-epoch', result[-1], epoch)
            writer.add_scalar('Metric/acc-cost', result[-1], accu_cost)
            args.logger.add_value('accuracy', result[-1])
            args.logger.add_value('epoch', epoch)
            args.logger.add_value('cmu-cost', accu_cost)
            valid_regions = np.array(buffer['valid_region'])
            args.logger.add_value('returned_eps', valid_regions)
            if buffer['escape_epoch'] is not None:
                args.logger.add_value('escape_epoch', buffer['escape_epoch'])
            print(f'valid regions on clients at epoch {epoch}: {valid_regions}')
            
            result_name = os.path.join(args.train_dir, 'result.tar')
            torch.save({
                'result': args.logger.dump()
            }, result_name)

        if args.save_model and epoch % args.save_interval == 1 and epoch != 1:
            file_name = os.path.join(args.train_dir, 'model_%04d.tar'%epoch )
            img_syn = torch.cat([workers[i].image_syn.detach().clone().cpu() for i in range(args.n_client)], 0)
            lbl_syn = torch.cat([workers[i].label_syn.detach().clone().cpu() for i in range(args.n_client)], 0)

            torch.save({
                'args': vars(args),
                'epoch': epoch,
                'img_syn': img_syn.cpu(),
                'lbl_syn': lbl_syn.cpu(),
                'state_dict': model_server.state_dict(),
            }, file_name)

    if (args.save_model):
        file_name = os.path.join(args.train_dir, 'model_last.tar')
        result_name = os.path.join(args.train_dir, 'result.tar')
        img_syn = torch.cat([workers[i].image_syn.detach().clone().cpu() for i in range(args.n_client)], 0)
        lbl_syn = torch.cat([workers[i].label_syn.detach().clone().cpu() for i in range(args.n_client)], 0)

        torch.save({
            'result': args.logger.dump()
        }, result_name)

        torch.save({
                'args': vars(args),
                'epoch': epoch,
                'img_syn': img_syn.cpu(),
                'lbl_syn': lbl_syn.cpu(),
                'state_dict': model_server.state_dict(),
            }, file_name)
    writer.close()

if __name__ == '__main__':
    args = parse_args()

    main(args)
