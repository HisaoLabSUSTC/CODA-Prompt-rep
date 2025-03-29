from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function

class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        super(Prompt, self).__init__(learner_config)

    def update_model(self, inputs, targets):

        # logits
        logits, prompt_loss = self.model(inputs, train=True)
        logits = logits[:,:self.valid_out_dim]

        # ce with heuristic
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    # sets model optimizers
    def init_optimizer_origin(self):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
        print('*****************************************')
        print(f'num parameters: {sum(p.numel() for p in params_to_opt if p.requires_grad)}')
        optimizer_arg = {'params': params_to_opt,
                         'lr': self.config['lr'],
                         'weight_decay': self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD', 'RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'], 0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)

        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    # sets model optimizers
    def init_optimizer(self, target=None, schedule=None):
        if schedule is None:
            schedule = self.schedule

        if type(self.config['lr']) is float:
            lr = self.config['lr']
        else:
            lr = self.config['lr'][0]

        lr_decreace_ratio = self.config['lr_decreace_ratio']
        larger_prompt_lr = self.config['args'].larger_prompt_lr

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            last = self.model.module.last
            prompt = self.model.module.prompt
        else:
            last = self.model.last
            prompt = self.model.prompt

        params_to_opt_p, names_p = [], []
        params_to_opt_l, names_l = [], []
        if self.config['mode'] in ['sys', 'pro', 'sub', 'non', 'noc']:
            # if fewshot testing self.config['mode'], only learn classifier: model.last
            for k, p in self.model.named_parameters():
                if 'last' in k and p.requires_grad:
                    params_to_opt_l.append(p)
                    names_l.append(k)
        elif target == 'last':
            for k, p in self.model.named_parameters():
                if 'last' in k and p.requires_grad:
                    params_to_opt_l.append(p)
                    names_l.append(k)
        elif target == 'prompt':
            for k, p in self.model.named_parameters():
                if 'prompt' in k and p.requires_grad:
                    params_to_opt_p.append(p)
                    names_p.append(k)
        elif target == 'p':
            for k, p in self.model.named_parameters():
                if 'e_p_' in k and p.requires_grad:
                    params_to_opt_p.append(p)
                    names_p.append(k)
                elif 'last' in k and p.requires_grad:
                    params_to_opt_l.append(p)
                    names_l.append(k)
        elif target == 'ka':
            for k, p in self.model.named_parameters():
                if ('e_k_' in k or 'e_a_' in k) and p.requires_grad:
                    params_to_opt_p.append(p)
                    names_p.append(k)
                elif 'last' in k and p.requires_grad:
                    params_to_opt_l.append(p)
                    names_l.append(k)
        else:
            for k, p in self.model.named_parameters():
                if 'prompt' in k and p.requires_grad:
                    params_to_opt_p.append(p)
                    names_p.append(k)
                elif 'last' in k and p.requires_grad:
                    params_to_opt_l.append(p)
                    names_l.append(k)

        print('******************* init optimizer **********************')
        print(f'optimizer params: {"all" if target is None else target} '
              f'len {[len(params_to_opt_p), len(params_to_opt_l)]}')
        print(f'[{sum(p.numel() for p in params_to_opt_p)}]: {names_p}')
        print(f'[{sum(p.numel() for p in params_to_opt_l)}]: {names_l}')

        if larger_prompt_lr:
            lrs = [lr, 0.1 * lr]
        else:       #
            lrs = [lr_decreace_ratio * lr, lr]
        params = [params_to_opt_p, params_to_opt_l]
        print(f'lrs: {lrs}')

        opt_args = []
        for idx in range(len(lrs)):
            _lr = lrs[idx]
            _params = params[idx]
            if len(_params) > 0:
                optimizer_arg = {'params':_params,
                                 'lr':_lr,
                                 'weight_decay':self.config['weight_decay']}
                if self.config['optimizer'] in ['SGD','RMSprop']:
                    optimizer_arg['momentum'] = self.config['momentum']
                elif self.config['optimizer'] in ['Rprop']:
                    optimizer_arg.pop('weight_decay')
                elif self.config['optimizer'] == 'amsgrad':
                    optimizer_arg['amsgrad'] = True
                    self.config['optimizer'] = 'Adam'
                elif self.config['optimizer'] == 'Adam':
                    optimizer_arg['betas'] = (self.config['momentum'],0.999)
                opt_args.append(optimizer_arg)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](opt_args, lr=lr)    # default lr
        # self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)

        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=schedule, gamma=0.1)
        else:       # no change
            self.scheduler = type('empty_scheduler', (), {})()
            self.scheduler.step = lambda x=0: None       # empty object scheduler with empty step() func.

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

# Our method!
class CODAPrompt(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda',prompt_param=self.prompt_param)
        return model

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):

    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual', prompt_param=self.prompt_param)
        return model

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):

    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param)
        return model