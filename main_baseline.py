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
from random import shuffle
from tqdm import tqdm

# import local library
from utils.fl_utils import set_model, update_model_global_optim, test, prepare_workers, loss_prox, compute_client_gradients
from utils.utils import Parser, LearningScheduler, FLLogger
from utils.misc import get_network, prepare_data, get_loops
from rdp_accountant import compute_sigma

from opacus.privacy_engine import PrivacyEngine

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', '--cfg', default=None, type=str, required=True)
    parser.add_argument('-seed', '--seed', default=None)

    parser.add_argument('-data-path', '--data-path', default='./datasets', type=str)
    parser.add_argument('-download', '--download', action='store_true')

    parser.add_argument('-save_path', '--save_path', default='./saves', type=str)

    # if start-epoch != 1, load the pretrained model
    parser.add_argument('-start-epoch', '--start-epoch', default=1, type=int)
    parser.add_argument('-start-model', '--start-model', default=None, type=str)
    parser.add_argument('-start-log', '--start-log', default=None, type=str)

    parser.add_argument('-verbose', '--verbose', action='store_true')
    parser.add_argument('-finetune', '--finetune', action='store_true')

    args = parser.parse_args()
    with open(args.cfg, 'r') as stream:
        settings = yaml.safe_load(stream)
    args = Parser(args, settings)
    args.name = os.path.basename(args.cfg).split('.')[0]

    if args.finetune:
        assert args.start_model is not None
        args.save_path = os.path.join(args.save_path, 'finetune')
        suffix = os.path.basename(args.start_model).split('.')[0]
        args.name = args.name + f'-{suffix}'

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

def train(args, global_optim, subnet_server, subnet, state_server, metric,
    device, workers, current_epoch, buffer, lr_scheduler, test_loader, warmup=False):
    subnet.train()

    client_samples = list(range(args.n_client))

    # buffers for standard training
    buffer['gradient_data'] = []
    buffer['gradient_rec1'] = []
    buffer['gradient_rec2'] = []
    buffer['gradient_rec3'] = []
    # buffers for dataset distillation methods
    buffer['dsc_images'] = []

    shuffle(client_samples)

    for id_client in client_samples[:args.n_update_client]:
        current_worker = workers[id_client]
        current_data_loader = current_worker.loader
        ### Initialize for DP
        if args.enable_privacy:
            iters = args.iteration * args.outer_loop * args.batch_loop
            if args.noise_multiplier is None:
                noise_multiplier = compute_sigma(args.target_epsilon, current_worker.sampling_rate, iters, args.target_delta)
            else:
                noise_multiplier = args.noise_multiplier

            subnet = copy.deepcopy(subnet_server)
            optimizer = optim.SGD(params=subnet.parameters(), lr=args['client_settings']['lr'],
                momentum=args['client_settings']['momentum'], weight_decay=args['client_settings']['weight_decay'])
            privacy_engine = PrivacyEngine(subnet, sample_rate=current_worker.sampling_rate, alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)), noise_multiplier=noise_multiplier, max_grad_norm=args.max_norm)
            privacy_engine.attach(optimizer)
        else:
            optimizer = current_worker.opt

            # mimic sending model weights to clients
            start_time = time.time()
            set_model(subnet_server, subnet.module, args)
            #print("--- %s seconds for copy submodel---" % (time.time() - start_time))

        if not warmup:
            lr_scheduler.set_opt(optimizer)
        for epoch_client in range(args.epoch_client):
            epoch_time = time.time()
            for batch_idx, (data, target) in enumerate(current_data_loader): # <-- now it is a distributed dataset
                data, target = data.to(device), target.to(device)
                
                output = subnet(data)
                if args.optimization == 'fedprox':
                    if args.enable_privacy:
                        # the privacy engine doesn't support parallel computing
                        loss = metric(output, target) + args.mu_loss_prox * loss_prox(subnet_server , subnet, device)
                    else:
                        loss = metric(output, target) + args.mu_loss_prox * loss_prox(subnet_server , subnet.module, device)
                else:
                    loss = metric(output, target)

                # if loss < 10:
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                #print("--- %s seconds for one training---" % (time.time() - start_time))

                if batch_idx % args.log_interval == 0:
                    for param_group in optimizer.param_groups:
                        current_learning_rate = param_group['lr']

                    print('Train Epoch: {}, Client: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {:.4f}'.format(
                        current_epoch, id_client, batch_idx * args.batch_size, len(current_data_loader) * args.batch_size,
                        100. * batch_idx / len(current_data_loader) / 100, loss.item(), current_learning_rate ))
                print("--- %s seconds for one local epoch---" % (time.time() - epoch_time))

                if args.optimization == 'fedsgd':
                    assert args.epoch_client == 1
                    break
            
        #start_time = time.time()
        if args.enable_privacy:
            compute_client_gradients(subnet_server, subnet, buffer, args)
        else:
            compute_client_gradients(subnet_server, subnet.module, buffer, args)

    update_model_global_optim(global_optim['optim'], subnet_server, buffer, test_loader, device, metric, current_epoch, args)

    if not warmup:
        lr_scheduler.step()

def create_server_opt(subnet_server, args):
    global_optim = {}
    if args.optimization == 'fedadam':
        global_optim['optim'] = optim.Adam(params=subnet_server.parameters(), lr=args.lr_net)
    elif args.optimization == 'fedlap':
        global_optim['optim'] = optim.SGD(params=subnet_server.parameters(), lr=args.lr_net)
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

    if args.start_model is not None:
        model_load_tmp = torch.load(args.start_model)
        model_server.load_state_dict(model_load_tmp["state_dict"] , strict=False)
        tmp_result = []
        test(args, model_server, device, test_loader, tmp_result)

    print(model_server)
    
    metric = nn.CrossEntropyLoss()
    args.logger = FLLogger(args, model=model_server)

    if args.start_log is not None:
        result_load_tmp = torch.load(args.start_log)['result']
        args.logger.load(result_load_tmp, epochs=args.start_epoch)

    subnet_server = model_server

    global_optim = create_server_opt(subnet_server, args)
        
    subnet = torch.nn.DataParallel(copy.deepcopy(subnet_server), device_ids=[0])

    # initialize worker on every client
    for i in range(args.n_client):
        workers[i].set_opt(optim.SGD(params=subnet.parameters(), lr=args['client_settings']['lr'],
           momentum=args['client_settings']['momentum'], weight_decay=args['client_settings']['weight_decay']))

    lr_scheduler = LearningScheduler(args)

    # log
    writer = SummaryWriter(args.log_dir)

    result = []
    accu_cost = 0

    for epoch in tqdm(range(args.start_epoch, args.epochs + 1)):
        sys.stdout.flush()

        # record communication cost
        cur_cost = 0
        for parameter in model_server.parameters(): cur_cost += parameter.nelement()
        
        # megabytes
        accu_cost += args.n_update_client * (cur_cost*4/1000/1000)

        buffer = {}
        train(args, global_optim, subnet_server, subnet, state_server, metric, device, workers, epoch, buffer, lr_scheduler, test_loader)
        if epoch % args.test_interval == 0:
            #test(args, model_server, device, test_loader, result)
            start_time = time.time()
            test(args, model_server, device, test_loader, result)
            print("--- %s seconds for test---" % (time.time() - start_time))
            writer.add_scalar('Metric/acc-epoch', result[-1], epoch)
            writer.add_scalar('Metric/acc-cost', result[-1], accu_cost)
            args.logger.add_value('accuracy', result[-1])
            args.logger.add_value('epoch', epoch)
            args.logger.add_value('cmu-cost', accu_cost)

        if args.save_model and epoch % args.save_interval == 1 and epoch != 1:
            file_name = os.path.join(args.train_dir, 'model_%04d.tar'%epoch )
            res = torch.from_numpy(np.array(result))

            torch.save({
                'args': vars(args),
                'epoch': epoch,
                'state_dict': model_server.state_dict(),
                'result': args.logger.dump()
            }, file_name)

    if (args.save_model):
        file_name = os.path.join(args.train_dir, 'model_last.tar')
        res = torch.from_numpy(np.array(result))

        torch.save({
                'args': vars(args),
                'epoch': epoch,
                'state_dict': model_server.state_dict(),
                'result': args.logger.dump()
            }, file_name)
    writer.close()

if __name__ == '__main__':
    args = parse_args()

    main(args)
