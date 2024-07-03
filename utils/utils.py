import argparse
import copy
import os
import torch
import torch.nn.functional as F
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

from collections import defaultdict
from scipy.special import softmax

from torch.utils.data import Dataset

from PIL import Image

from utils.augmentation import ParamDiffAug

choice_dict = {}
choice_dict['strategy'] = ['baseline']
choice_dict['dataset'] = ['CIFAR100', 'CIFAR10', 'imagenet', 'MNIST', 'FashionMNIST']
choice_dict['optimization'] = ['fedavg', 'fedprox', 'fedadam', 'fedlap', 'fedsgd']

default_dict =  {}
default_dict['optimization'] = 'fedavg'
default_dict['truncate'] = None
default_dict['noise_multiplier'] = None
default_dict['reg_type'] = None
default_dict['reg_weight'] = 0
default_dict['non_balanced'] = False
default_dict['reinit_image'] = False

class Parser(dict):
    def __init__(self, *args):
        super(Parser, self).__init__()
        for d in args:
            if isinstance(d, argparse.Namespace):
                d = vars(d)
            for k, v in d.items():
                if k == 'seed' and k in self.keys() and self[k] != None:
                    print(f'{k} is found in arg parser.')
                    continue
                assert k not in self.keys() or k == 'seed', f'{k}'
                k = k.replace('-', '_')
                #check whether arguments match the limited choices
                if k in choice_dict.keys() and v not in choice_dict[k]:
                    raise ValueError(f'Illegal argument \'{k}\' for choices {choice_dict[k]}')
                # convert string None to Nonetype, which is a side effect of using yaml
                self[k] = None if v == 'None' else v

        # check whether the default options has been in args; otherswise, add it.
        for k in default_dict.keys():
            if k not in self.keys():
                self[k] = default_dict[k] 

        # Parse parameters when the DSA strategy is enabled
        if 'fedlap' in self['optimization'] and self['dsa']:
            self['dsa_param'] = ParamDiffAug()

        # temp. buffer used for passing arguments across functions, typically used for logging
        self['tmp_buffer'] = {}

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, key, val):
        self[key] = val

    def clean_buf(self):
        self['tmp_buffer'] = {}

class LearningScheduler(object):
    def __init__(self, args):
        kwargs = args.client_settings
        self.kwargs = kwargs
        self.args = args
        self.type = kwargs['type']
        dummy_opt = torch.optim.SGD(torch.nn.Linear(1,1).parameters(), lr=kwargs['lr'])
        self.manual_function = None
        if self.type == 'multistep':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                dummy_opt, milestones=kwargs['milestones'], gamma=kwargs['gamma'])
        elif self.type == 'cosine_restart':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(dummy_opt, 
                T_0=kwargs['T_0'], T_mult=kwargs['T_mult'], eta_min=kwargs['eta_min'])
        elif self.type == 'cosine_decay':
            self.manual_function = self._cosine_decay
        elif self.type == 'constant':
            self.manual_function = self._constant
        elif self.type == 'piecewise_constant':
            self.manual_function = self._piecewise_constant
        else:
            raise NotImplementedError(f'Unknown lr scheduler {self.type}')

        self.opt = None
        self.step_cnt = 0
        self.milestone = kwargs['milestones']

    def set_opt(self, opt):
        self.opt = opt
        lr = self.get_lr()
        for g in self.opt.param_groups:
            g['lr'] = lr

    def step(self):
        assert self.opt is not None
        
        self.step_cnt +=1

        if self.manual_function is None:
            self.lr_scheduler.step()
        for g in self.opt.param_groups:
            g['lr'] = self.get_lr()

    def get_lr(self):
        if self.manual_function is None: 
            return self.lr_scheduler.get_last_lr()[0]
        else:
            return self.manual_function()

    def _cosine_decay(self):
        return max(self.kwargs['lr'] * (1 + np.cos(np.pi * (self.step_cnt-1) / (self.args.epochs-1) ) ) / 2 , 1e-6)

    def _constant(self):
        return self.kwargs['lr']

    def _piecewise_constant(self):
        break_point = self.args.epochs // 2 - self.args.epochs // (2 * self.args.num_stages)
        remaining_epochs = self.args.epochs - break_point
        if self.step_cnt < break_point:
            return self.kwargs['lr']
        else:
            return max(self.kwargs['lr'] * (1 + np.cos(np.pi * (self.step_cnt-1) / (self.args.epochs-1) ) ) / 2 , 1e-6)

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

class FLLogger():
    def __init__(self, args, **kwargs):
        #super().__init__(args, **kwargs)
        self.args = args
        self.records = defaultdict(list)

    def add_value(self, keys, val):
        key_split = keys.split('/')
        map = self.recurssive_fetch(key_split, self.records, 0)
        assert isinstance(map, list), f'the entity of key {keys} is a {type(map)}'
        map.append(val)

    # e.g., clients/5/eval_loss
    def recurssive_fetch(self, keys, cur_map, index):
        if index == len(keys) - 1:
            return cur_map[keys[index]]
        
        if not isinstance(cur_map[keys[index]], defaultdict):
            assert len(cur_map[keys[index]]) == 0, f'the entity is not empty with length {len(cur_map[keys[index]])} and the key {keys[:index+1]}'
            cur_map[keys[index]] = defaultdict(list)
        # proceed
        return self.recurssive_fetch(keys, cur_map[keys[index]], index+1)

    def dump(self):
        return self.records

    def load(self, d, epochs=None):
        self.records = d
        if epochs is not None:
            for k in self.records.keys():
                self.records[k] = self.records[k][:epochs]
        
def compute_model_diff(m_new, m_old):
    with torch.no_grad():
        concat_weight_new = [p.detach().clone().view(-1) for p in m_new.parameters()]
        concat_weight_old = [p.detach().clone().view(-1) for p in m_old.parameters()]

        concat_weight_new = torch.cat(concat_weight_new, 0)
        concat_weight_old = torch.cat(concat_weight_old, 0)

        eps = torch.norm(concat_weight_new - concat_weight_old, p=2)

    return eps.cpu().numpy()
