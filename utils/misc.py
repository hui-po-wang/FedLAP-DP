import os
import time
import numpy as np
import yaml
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

from random import shuffle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from .networks import MLP, ConvNet, LeNet, AlexNet, VGG11BN, VGG11, ResNet18, ResNet18BN_AP

__all__ = ['get_dataset', 'get_network', 'get_time', 'get_loops', 'get_daparam', 'get_eval_pool', 'mkdir', 'flatten_tensor', 'inf_train_gen', 'TensorDataset', 'load_yaml', 'write_yaml']

def prepare_data(args, use_cuda, state_server, all_train=False):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    split_in = False
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    order_list = []
    if args.noniid and (args.dataset in ['MNIST', 'FashionMNIST', 'CIFAR10']):
        split_in = True
        order_list = [5,6,1,2,0,8,9,3,4,7]
    elif args.noniid and args.dataset in ['CIFAR100']:
        split_in = True
        order_list = [88, 17, 81, 18, 11, 62, 82, 54, 96, 93, 92, 46, 99, 20,  5,  2, 70,
            86, 73, 69, 33, 25, 71, 55, 13, 89, 40, 24, 67, 63, 19, 75, 95, 35,
            58, 39, 65, 85, 94,  0, 77, 34, 47, 66, 59,  4, 97, 21, 49, 83, 98,
            42, 53, 76, 26, 51,  9, 12, 50, 74,  8, 60, 45, 52,  3, 27, 31,  7,
            84, 38, 43, 56, 79, 64, 41,  1,  6, 10, 48, 78, 37, 14, 57, 23, 28,
            22, 87, 16, 36, 61, 90, 72, 29, 32, 80, 44, 30, 15, 68, 91]
    elif args.noniid:
        raise NotImplementedError()

    X_train_total = dst_train.data
    Y_train_total = dst_train.targets
    if isinstance(Y_train_total, list):
        Y_train_total = np.array(Y_train_total)
    elif isinstance(Y_train_total, torch.Tensor):
        Y_train_total = Y_train_total.numpy()
    order = np.array(order_list)
    n_class_per_client = int(num_classes / args.n_client)

    trainset = []
    if args.noniid:
        if args.non_balanced:
            for i_n in range(args.n_client):
                if i_n != args.n_client - 1:
                    if i_n % 2 == 0:
                        indices_train_10 = np.array([i in order[i_n * n_class_per_client: (i_n + 1) * n_class_per_client + 1] for i in Y_train_total])
                    else:
                        indices_train_10 = np.array([i in order[i_n * n_class_per_client: (i_n + 1) * n_class_per_client - 1] for i in Y_train_total])

                else:
                    indices_train_10 = np.array([i in order[i_n * n_class_per_client: (i_n + 1) * n_class_per_client] for i in Y_train_total])
                X_train = X_train_total[indices_train_10]
                Y_train = Y_train_total[indices_train_10]
                trainset.append(TensorDataset(X_train, Y_train, transform=dst_train.transform))
        else:
            for i_n in range(args.n_client):
                indices_train_10 = np.array([i in order[i_n * n_class_per_client: (i_n + 1) * n_class_per_client] for i in Y_train_total])
                X_train = X_train_total[indices_train_10]
                Y_train = Y_train_total[indices_train_10]
                trainset.append(TensorDataset(X_train, Y_train, transform=dst_train.transform))
    else:
        for i_n in range(args.n_client):
                indices_train_10 = list(range(len(X_train_total)))
                shuffle(indices_train_10)
                indices_train_10 = np.array(indices_train_10[:(len(X_train_total) // args.n_client)])

                X_train = X_train_total[indices_train_10]
                Y_train = Y_train_total[indices_train_10]
                trainset.append(TensorDataset(X_train, Y_train, transform=dst_train.transform))

    state_server['num_classes'] = num_classes
    state_server['channel'] = channel
    state_server['im_size'] = im_size

    args.num_classes = num_classes
    args.channel = channel
    args.im_size = im_size

    if all_train:
        return trainset, dst_train, testloader, split_in    
    else:
        return trainset, testloader, split_in
    
def get_dataset(dataset, data_path):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'CelebA':
        channel = 3
        im_size = (64, 64)
        num_classes = 2
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        target_transform = lambda arr: arr[20]  # obtain the value of the 'Male' attribute
        dst_train = datasets.CelebA(data_path, split='train', download=True, transform=transform, target_transform=target_transform)  # no augmentation
        dst_test = datasets.CelebA(data_path, split='test', download=True, transform=transform, target_transform=target_transform)
        class_names = ['female', 'male']

    else:
        exit('unknown dataset: %s' % dataset)

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=2)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader


def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


def get_network(model, channel, num_classes, im_size=(32, 32)):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes, im_size=im_size)
    elif model == 'MLPIN':
        net = MLP(channel=channel, num_classes=num_classes, net_norm='instancenorm', im_size=im_size)
    elif model == 'MLPBN':
        net = MLP(channel=channel, num_classes=num_classes, net_norm='batchnorm', im_size=im_size)
    elif model == 'MLPLN':
        net = MLP(channel=channel, num_classes=num_classes, net_norm='layernorm', im_size=im_size)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none')
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling')
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling')

    else:
        net = None
        exit('unknown model: %s' % model)

    gpu_num = torch.cuda.device_count()
    if gpu_num > 0:
        device = 'cuda'
        if gpu_num > 1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    elif ipc == 100:
        outer_loop, inner_loop = 50, 10
    elif ipc == 150:
        outer_loop, inner_loop = 50, 10
    elif ipc == 200:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc' % ipc)
    return outer_loop, inner_loop


def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if 'MNIST' in dataset:
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']:  # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M':  # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'W':  # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D':  # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A':  # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL']
    elif eval_mode == 'P':  # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N':  # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S':  # itself
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


def mkdir(path):
    '''make dir if not exist'''
    if not os.path.exists(path):
        os.makedirs(path)


def flatten_tensor(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param


def inf_train_gen(trainloader):
    while True:
        for images, targets in trainloader:
            yield (images, targets)


class TensorDataset(Dataset):
    def __init__(self, images, labels, transform=None):  # images: n x c x h x w tensor
        self.images = images.numpy() if isinstance(images, torch.Tensor) else images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        label = int(self.labels[index]) if isinstance(self.labels, torch.Tensor) else self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.images.shape[0]


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data


def write_yaml(data, filepath):
    with open(filepath, 'w') as f:
        yaml.dump(data, f)

