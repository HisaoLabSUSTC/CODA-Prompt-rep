from __future__ import print_function
import sys
import math
from typing import Optional, Union, Tuple, Dict, Any, List
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor, LongTensor
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
import copy
import torchvision
from torch.autograd import Variable, Function
import pandas as pd
from datetime import datetime
from scipy.optimize import linear_sum_assignment

from .default import NormalNN, weight_reset, accumulate_acc
from .prompt import Prompt
from utils.schedulers import CosineSchedule
from .pmo_utils import Pool, Mixer, available_setting, task_to_device, cal_hv_weights, normalize
from models.losses import prototype_loss
from mo_optimizers.functions_evaluation import fastNonDominatedSort
import dataloaders

from sklearn.cluster import k_means, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score


# Our PMO (Pool & Multi-Objective)
class SLOTPrompt(Prompt):
    def __init__(self, learner_config):
        super(SLOTPrompt, self).__init__(learner_config)
        # self.pool_size = self.prompt_param[1][3]        # =0 if do not enable pool hv loss
        # self.pool = Pool(self.pool_size, self.seed)

        self.train_dataset = None
        self.t = 0
        self.epoch = 0
        self.epochs = 0     # total epoch in this task

        # load aux
        # aux_dataset = dataloaders.CGQA(
        #     self.config['aux_root'],
        #     train=False, validation=True, download_flag=False, seed=self.config['seed'])
        # aux_dataset.load_dataset(9, train=False)   # consider all samples: 100 classes with 5000 samples.
        # self.aux = Auxiliary(aux_dataset)

        if self.config['args'].use_old_samples_for_reg:
            self.aux = Auxiliary(self.config['args'], None, self.tasks)

        config = self.config['prompt_param'][1]
        while len(config) < 15:
            config.append(0)

        try:
            prompt = self.model.module.prompt
        except:
            prompt = self.model.prompt
        self.e_layers = prompt.e_layers
        self.FPS = prompt.FPS
        # self.e_pool_size = prompt.e_pool_size       # 100
        self.key_d = prompt.key_d                   # 64

        # s2p regularizer
        self.s2p_copy = None

    def create_model(self):
        cfg = self.config
        n_task = self.prompt_param[0]
        tasks = cfg['tasks']
        prompt_parameters = self.prompt_param[1][:2]    # [100, 8]
        prompt_parameters.append(self.config['args'].n_slots)
        prompt_parameters.append(self.config['args'].n_iters)
        prompt_parameters.append(self.config['args'].slot_temp)
        prompt_parameters.append(self.config['args'].s2p_temp)
        prompt_parameters.append(self.config['args'].s2p_mode)

        prompt_param = [[n_task, tasks], prompt_parameters]
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'slot',prompt_param=prompt_param, use_vit_emb=False)

        # model.prompt.expert_predictor[0] = nn.Sequential(
        #     nn.Linear(len(cfg['tasks'][0]), len(cfg['tasks'][0])),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(len(cfg['tasks'][0]), 2))

        return model

    def load_model(self, filename, drop_last=False, task_id=-1, if_from_outside=False, slot_pre_learn_model='none',
                   freeze=False):
        flag = False        # True if need to further train
        # if slot_pre_learn_model is not none, load slots
        if slot_pre_learn_model != 'none':
            filename = ('/'.join(self.config['log_dir'].split('/')[:-1]) + '/' +
                        slot_pre_learn_model + f'/models/repeat-{self.seed+1}/task-{task_id+1}/')
            print(f'redirect loading slot model from {filename}.')
            try:
                state_dict = torch.load(filename + 'class.pth')
            except:
                filename = ('/'.join(self.config['log_dir'].split('/')[:-1]) + '/' +
                            slot_pre_learn_model + f'/models/repeat-{self.seed+1}/task-1/')
                print(f'WARNING, donot find model file, assuming a MT model, redirect loading from {filename}')
                state_dict = torch.load(filename + 'class.pth')
            # complete with/without module and collect slot
            for key in list(state_dict.keys()):
                if 'slot_attn' in key:
                    if 'module' in key:
                        state_dict[key[7:]] = state_dict[key]
                    else:
                        state_dict[f'module.{key}'] = state_dict[key]
                else:
                    del state_dict[f'{key}']
            self.model.load_state_dict(state_dict, strict=False)
            self.log(f'=> Load Done with params {list(state_dict.keys())}')

            names = []
            for k, p in self.model.named_parameters():
                if 'slot_attn' in k:
                    p.requires_grad = False
                    names.append(k)
            self.log(f'=> Freeze slot model: {names}')

        else:
            ## from_outside to enable load pretrained model for the 1-st task.
            # # random init pool
            # if self.pool is None:
            #     self.register_buffer('pool', torch.randn(self.e_pool_size, self.key_d).float())

            if self.t == 0 and if_from_outside and self.config['t0_model_from'] != 'none':     # 1-st task load from warm-started one
                filename = ('/'.join(self.config['log_dir'].split('/')[:-1]) + '/' +
                            self.config['t0_model_from'] + f'/models/repeat-{self.seed+1}/task-1/')
                print(f'redirect loading model from {filename}.')

            state_dict = torch.load(filename + 'class.pth')
            # complete with/without module.
            for key in list(state_dict.keys()):
                if 'module' in key:
                    state_dict[key[7:]] = state_dict[key]
                else:
                    state_dict[f'module.{key}'] = state_dict[key]
            if drop_last:
                del state_dict['module.last.weight']; del state_dict['module.last.bias']
                del state_dict['last.weight']; del state_dict['last.bias']
                # if 'module.last.weight' in state_dict:
                #     del state_dict['module.last.weight']; del state_dict['module.last.bias']
                # else:
                #     del state_dict['last.weight']; del state_dict['last.bias']
                # self.model.load_state_dict(state_dict, strict=False)

            for k in list(state_dict.keys()):
                if 'e_' in k and 's2p' not in k:        # if load CODA's PKA
                    idx = k.index('e_')
                    new_k = k[:idx] + 's2p.' + k[idx:]
                    state_dict[new_k] = state_dict[k]

            self.model.load_state_dict(state_dict, strict=False)
            self.log(f'=> Load Done')
            # self.log(f'=> Load Done with params {list(state_dict.keys())}')

        if freeze:
            self.log('=> Freeze backbone')     # on CFST
            for k, p in self.model.named_parameters():
                if 'last' not in k:
                    p.requires_grad = False

        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

        # if flag:
        #     # freeze prompt:[except expert_predictor] and last
        #     for k, p in self.model.named_parameters():
        #         if 'expert_predictor' not in k:
        #             p.requires_grad = False

        return flag

    # sets model optimizers
    def init_optimizer(self, t=0, target=None, schedule=None):
        """target - slot; prompt+reuse(new); last"""
        if schedule is None:
            schedule = [30]
        if target is None:
            target = 'all'
        params_to_opt_s, names_s = [], []
        params_to_opt_p, names_p = [], []
        params_to_opt_l, names_l = [], []
        for k, p in self.model.named_parameters():
            if p.requires_grad:
                if 'last' in target and 'last' in k:
                    # include CFST self.config['mode'] in ['sys', 'pro', 'sub', 'non', 'noc']:
                    # if fewshot testing self.config['mode'], only learn classifier: model.last
                    params_to_opt_l.append(p)
                    names_l.append(k)
                if 'slot' in target and 'slot_attn' in k:
                    params_to_opt_s.append(p)
                    names_s.append(k)
                if 'prompt' in target and 'reuse' in target and (
                        'prompt' in k and 'e_p' not in k and 'slot_attn' not in k):     # reuse do not train p
                    params_to_opt_p.append(p)
                    names_p.append(k)
                if 'prompt' in target and 'new' in target and ('prompt' in k and 'slot_attn' not in k):
                    params_to_opt_p.append(p)
                    names_p.append(k)
                if target == 'all':      # all things train simultaneously
                    if 'slot_attn' in k:
                        params_to_opt_s.append(p)
                        names_s.append(k)
                    if 'prompt' in k and 'slot_attn' not in k:
                        params_to_opt_p.append(p)
                        names_p.append(k)
                    elif 'last' in k:
                        params_to_opt_l.append(p)
                        names_l.append(k)

        print('******************* init optimizer **********************')
        print(f'optimizer params: {"all" if target is None else target} '
              f'len {[len(params_to_opt_s), len(params_to_opt_p), len(params_to_opt_l)]}')
        print(f'slots[{sum(p.numel() for p in params_to_opt_s)}]: {names_s}')
        print(f'prompt[{sum(p.numel() for p in params_to_opt_p)}]: {names_p}')
        print(f'last[{sum(p.numel() for p in params_to_opt_l)}]: {names_l}')

        lr = self.config['lr']          # [1e-3, 1e-3,...]
        slot_lr = self.config['slot_lr']    # [1e-4, 1e-4,...]
        if type(lr) is list and t >= len(lr):
            lr = lr[-1]
        elif type(lr) is list:
            lr = lr[t]
        if type(slot_lr) is list and t >= len(slot_lr):
            slot_lr = slot_lr[-1]
        elif type(slot_lr) is list:
            slot_lr = slot_lr[t]
        # slot_lr=1e-4; lr=1e-3

        lr_decreace_ratio = self.config['lr_decreace_ratio']   # for prompt
        larger_prompt_lr = self.config['args'].larger_prompt_lr

        if larger_prompt_lr:
            lrs = [slot_lr, lr, 0.1 * lr]
        else:       #
            lrs = [slot_lr, lr_decreace_ratio * lr, lr]
        print(f'lrs: {lrs}')

        opt_args = []
        params = [params_to_opt_s, params_to_opt_p, params_to_opt_l]
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
        if target == 'slot':        # only learn slot in this phase
            schedule_type = self.config['slot_schedule_type']     # cosann
        else:
            schedule_type = self.schedule_type

        if schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=schedule[-1])
        elif schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=schedule, gamma=0.1)
        elif schedule_type == 'cosann':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=int(schedule[-1]/2), T_mult=1, eta_min=lr/100)
        else:       # no change
            self.scheduler = type('empty_scheduler', (), {})()
            self.scheduler.step = lambda x=0: None       # empty object scheduler with empty step() func.

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        self.init_train_log()

        self.train_dataset = train_dataset
        self.t = train_dataset.t
        self.n_cls = len(self.tasks[self.t])     # tasks: [[0,1,...,49], [50,...,59], ...]
        print(f'num of classes: {self.n_cls}.')
        # try to load model
        need_train = True
        if not self.overwrite:
            # load slot model if specified
            if model_save_dir is not None and self.config['slot_pre_learn_model'] != 'none':
                # raise exp if no slot trained but train prompt
                # model_save_dir will be None if do compositional few-shot testing and model is load in trainer_ft.py
                self.load_model(model_save_dir, task_id=self.t,
                                slot_pre_learn_model=self.config['slot_pre_learn_model'])
            try:
                flag = self.load_model(model_save_dir, task_id=self.t, if_from_outside=True)
                need_train = flag
            except:
                pass

        # data weighting
        self.data_weighting(train_dataset)
        if need_train:
            # backup slot2prompt mapping
            if self.t > 0:
                try:
                    model = self.model.module
                except:
                    model = self.model

                if self.config['args'].use_weight_reg:
                    self.log(f'record s2p weights')
                    self.s2p_copy = copy.deepcopy(model.prompt.s2p)
                else:
                    self.s2p_copy = None

            # Determine schedule list for this task
            schedule = self.schedule     # T*[[epochs for slots], [enhance reuse], [learn new prompt]] or []
            if self.t >= len(schedule):
                schedule = schedule[-1]
            else:
                schedule = schedule[self.t]   # [[epochs for slots], [enhance reuse], [learn new prompt]]

            if type(schedule) is not list:      # for CFST: 20
                schedule = [[schedule], [0], [schedule]]  # for CFST: [[20], [0], [20]]

            # determine train_phases for this task
            if self.config['only_learn_slot']:      # only learn slot
                optimizer_targets = ['slot']
                schedule_phases = [0]
            elif model_save_dir is None:        # cfst do not load model during training
                optimizer_targets = ['last']
                schedule_phases = [0]
            elif self.config['slot_pre_learn_model'] == 'none':    # learn slot and prompt
                # if schedule[0][-1] > 0:         # if learn slot separately.
                #     optimizer_targets = ['slot', 'prompt+reuse+last', 'prompt+new+last']
                #     schedule_phases = [0, 1, 2]
                # else:
                optimizer_targets = ['slot+prompt+reuse+last', 'slot+prompt+new+last']
                schedule_phases = [1, 2]
            elif self.config['slot_pre_learn_model'] != 'none':     # specify slot attn model, only learn prompt
                optimizer_targets = ['prompt+reuse+last', 'prompt+new+last']
                schedule_phases = [1, 2]
            else:
                raise Exception(f"Incorrect condition - "
                                f"only_learn_slot: {self.config['only_learn_slot']}, "
                                f"slot_pre_learn_model: {self.config['slot_pre_learn_model']}, "
                                f"model_save_dir: {model_save_dir}.")
            self.log(f'Optimizer targets: {optimizer_targets}')

            for optimizer_target, schedule_phase in zip(optimizer_targets, schedule_phases):
                self.log(f'Phase：{optimizer_target}, schedule: {schedule[schedule_phase]}')

                # define tracking things
                res = dict()
                batch_timer = Timer()
                res['Loss'] = AverageMeter()
                res['Train Acc'] = AverageMeter()
                res['Time'] = AverageMeter()

                epochs = schedule[schedule_phase][-1]     # [,30] -> 30
                if epochs == 0:
                    print(f'Skip this phase, cause epochs=0')
                    continue

                self.epochs = epochs
                if self.reset_optimizer:  # Reset optimizer before learning each task
                    self.log('Optimizer is reset')
                    self.init_optimizer(t=self.t, target=optimizer_target, schedule=schedule[schedule_phase])

                for epoch in range(epochs):
                    self.epoch = epoch

                    if self.config['args'].use_old_samples_for_reg:
                        self.aux.update_source(train_dataset, self.t)  # aux samples from the current task

                    if epoch > 0: self.scheduler.step()
                    for param_group in self.optimizer.param_groups:
                        self.log('LR:', param_group['lr'])
                    batch_timer.tic()
                    for i, sample in enumerate(train_loader):
                        self.batch_idx = i

                        concepts = None
                        if hasattr(train_dataset, "return_concepts") and train_dataset.return_concepts:
                            x, y, concepts, task = sample
                        else:
                            x, y, task = sample

                        # verify in train mode
                        self.model.train()

                        # send data to gpu
                        if self.gpu:
                            x = x.cuda()
                            y = y.cuda()
                            # if concepts is not None:
                            #     concepts = concepts.cuda()      # [bs, 224, 224]
                            # task = task.cuda()

                        # # debug
                        # print(f'x shape: {x.shape}, y: {y}, task: {task}')

                        # model update
                        loss, output, loss_dict = self.update_model(x, y, optimizer_target=optimizer_target)

                        # measure elapsed time
                        res['Time'].update(batch_timer.toc())
                        batch_timer.tic()

                        # measure accuracy and record loss
                        y = y.detach()
                        accumulate_acc(output, y, task, res['Train Acc'], topk=(self.top_k,))
                        res['Loss'].update(loss, y.size(0))
                        # create or maintain records
                        for key, v in loss_dict.items():
                            visual_key = ''     # e.g., IntrConsLoss
                            for word in key.split('_'):
                                visual_key = visual_key + word[0].upper()+word[1:4].lower()
                            if visual_key not in res.keys():
                                res[visual_key] = AverageMeter()
                            res[visual_key].update(v, y.size(0))

                        batch_timer.tic()

                    # eval update
                    self.log(
                        'Epoch:{epoch:.0f}/{total:.0f}'.format(
                            epoch=self.epoch + 1, total=epochs))
                    log_str = (' * Loss {loss.avg:.3f} | '
                               'Train Acc {acc.avg:.3f} | ').format(loss=res['Loss'], acc=res['Train Acc'])
                    for name, meter in res.items():
                        if name != 'Loss' and name != 'Train Acc' and name != 'Time':
                            log_str = log_str + '{name} {meter.avg:.3f} | '.format(name=name, meter=meter)
                    log_str = log_str + 'Time {time:.3f}s ({i} batches)'.format(
                        time=res['Time'].avg*len(train_loader), i=len(train_loader),)
                    self.log(log_str)

                    # reset
                    for name in res.keys():
                        res[name] = AverageMeter()

                    # validation
                    if val_loader is not None:

                        if 'slot' == optimizer_target:      # this phase only learn slot with reconstruction task
                            val_recon_loss = self.validation(val_loader, slot_recon_loss=True)
                            # log
                            self.epoch_log['scaler']['Tag'].append(f'val_recon_loss')
                            self.epoch_log['scaler']['Idx'].append(self.epoch)
                            self.epoch_log['scaler']['Value'].append(val_recon_loss)
                        else:
                            val_acc = self.validation(val_loader)
                            # log
                            self.epoch_log['scaler']['Tag'].append(f'val_acc')
                            self.epoch_log['scaler']['Idx'].append(self.epoch)
                            self.epoch_log['scaler']['Value'].append(val_acc)

                    # if self.epoch % 10 == 0:
                    if self.epoch == 0:
                        '''nvidia-smi'''
                        os.system('nvidia-smi')

        if self.config['args'].use_feature_statistics:
            self.log(f'Phase III: update feature statistics for labels')
            self.collect_statistics(train_loader, train_dataset)
        if self.config['args'].use_slot_statistics:
            self.log(f'Phase III: update slot statistics for labels in task{self.t}')
            self.collect_slot_statistics(train_loader, train_dataset, save=True)
            # self.collect_slot_statistics_all(train_loader, train_dataset)
            # self.collect_slot_pool(train_loader, train_dataset)

        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        try:
            return res['Time'].avg
        except:
            return None

    def forward(self, inputs, targets, train=False, learn_slots=True, only_slots=False,
                prompt_phase='new', model=None):
        res = dict()
        if model is None:
            model = self.model
        # try:
        #     model = model.module
        # except:
        #     model = model
        q = model(inputs, obtain_q=True, learn_slots=learn_slots, train=train, prompt_phase=prompt_phase)
        # q = model.obtain_q(inputs, learn_slots=learn_slots, train=train, prompt_phase=prompt_phase)
        prompts, selections, slot_weights, w_slots, slots, attn, recon_loss = q
        # bs, t, k, e, p, d = prompts.shape    # [bs, t1, k1, e5, p8, d768]
        # bs, t, k, e, pp = selections.shape   # [bs, t1, k1, e5, pp100?]
        # bs, t, k = slot_weights.shape    # [bs, t1, k10]
        # bs, t, h = w_slots.shape    # [bs, t1, h128]    mapped slots @ weights
        # bs, t, k, h = slots.shape  # [bs, t1, k10, h128]
        res['prompts'] = prompts
        res['selections'] = selections
        res['slot_weights'] = slot_weights
        res['w_slots'] = w_slots
        res['slots'] = slots
        res['attn'] = attn
        res['recon_loss'] = recon_loss

        # slot-wise prompt
        bs, t, k, e, p, d = prompts.shape
        batched_prompts = prompts.reshape(bs, t * k, e, p, d)
        assert t == 1  # if not 1, should permutate t with e then reshape to t*k

        sum_prompts = torch.sum(batched_prompts, dim=1)  # [bs, e, p, d]

        # loss = torch.zeros(1).mean().to(prompts.device)
        # pen: penultimate features; train: same forward as batch training.
        if only_slots:
            with torch.no_grad():
                out, features = self.model(
                    inputs, q=sum_prompts,
                    pen=True, train=train,
                    # register_blk=hard_l,
                    debug_mode=self.debug_mode)
        else:
            out, features = self.model(
                inputs, q=sum_prompts,
                pen=True, train=train,
                # register_blk=hard_l,
                debug_mode=self.debug_mode)
        # features: [bs, 768]
        # out is logits: [bs, 100]

        res['logits'] = out
        res['features'] = features

        return res

    def update_model(self, inputs, targets, optimizer_target='slot+prompt+new+head', match_pool=False,
                     p=30, tau=3):
        collections = dict()        # collection for output
        self.optimizer.zero_grad()
        try:
            model = self.model.module
        except:
            model = self.model
        FPS = self.FPS

        prompt_phase = 'new' if 'new' in optimizer_target else 'reuse'
        res = self.forward(inputs, targets, train=True,
                           learn_slots='slot' in optimizer_target, only_slots='slot' == optimizer_target,
                           prompt_phase=prompt_phase)
        prompts = res['prompts']
        selections = res['selections']
        slot_weights = res['slot_weights']
        w_slots = res['w_slots']
        slots = res['slots']
        attn = res['attn']
        recon_loss = res['recon_loss']
        out = res['logits']     # [bs, 100]
        features = res['features']

        collections['min_slot_weights'] = torch.min(slot_weights).item()
        collections['max_slot_weights'] = torch.max(slot_weights).item()

        if self.debug_mode:
            print('samples:', inputs.shape, 'prompts:', prompts.shape)

        # q = model.obtain_q(inputs, learn_slots=learn_slots, train=True, prompt_phase=prompt_phase)
        # prompts, selections, ws, slots, attn, recon_loss = q
        # bs, t, k, e, p, d = prompts.shape
        # bs, t, k, h = slots.shape  # [bs, t1, k30, h128]

        # # mk
        # K = model.prompt.slot_attn_class_key        # [c100, h128]
        # s = self.last_valid_out_dim
        # mk_logit, mk_weights = self.forward_mk(slots.reshape(bs, t*k, h), K)
        #
        # mk_logit[:, :s] = -float('inf')
        # mk_loss = F.cross_entropy(mk_logit, targets.long())
        #
        # self.epoch_log['scaler']['Tag'].append('loss/mk_loss')
        # self.epoch_log['scaler']['Idx'].append(self.epoch)
        # self.epoch_log['scaler']['Value'].append(mk_loss.item())

        loss = torch.zeros(1).mean().to(out.device)

        if 'slot' in optimizer_target:
            # collect recon_loss
            recon_loss = torch.mean(torch.stack(recon_loss))       # list [1\T]

            if self.debug_mode:
                print(f'slot recon loss: {recon_loss.item()}')

            loss = loss + recon_loss

            self.epoch_log['scaler']['Tag'].append('loss/slot_recon_loss')
            self.epoch_log['scaler']['Idx'].append(self.epoch)
            self.epoch_log['scaler']['Value'].append(recon_loss.item())

            collections['slot_recon_loss'] = recon_loss.item()

            # positive samples to enhance intra-class consistency
            if self.config['args'].use_intra_consistency_reg:
                intra_consistency_loss = self._intra_consistency_reg(slots, slot_weights, w_slots, targets)
                loss = loss + self.config['args'].intra_consistency_reg_coeff * intra_consistency_loss

                self.epoch_log['scaler']['Tag'].append('loss/intra_consistency_loss')
                self.epoch_log['scaler']['Idx'].append(self.epoch)
                self.epoch_log['scaler']['Value'].append(intra_consistency_loss.item())

                collections['intra_consistency_loss'] = intra_consistency_loss.item()

            # image-wise mse for slot cosine sim vs I
            if self.config['args'].use_slot_ortho_reg:
                bs, t, k, h = slots.shape  # [bs, t1, k30, h128]
                img_slots = slots.reshape(bs, t*k, h)
                slot_ortho_reg_mode = self.config['args'].slot_ortho_reg_mode

                if 'dot' in slot_ortho_reg_mode:
                    sim = torch.einsum('bkh,bnh->bkn', img_slots, img_slots) * (
                            self.config['args'].slot_ortho_reg_temp * (h ** -0.5))
                    # [bs, k, k]
                else:
                    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                    sim = cos(img_slots.unsqueeze(1), img_slots.unsqueeze(2)
                              ) * self.config['args'].slot_ortho_reg_temp  # [bs, k, k]

                collections['min_slot_sim'] = torch.min(sim).item()
                collections['avg_slot_sim'] = torch.mean(sim).item()

                if 'l2' in slot_ortho_reg_mode:
                    eye = torch.eye(t*k).expand_as(sim).to(sim.device)
                    slot_ortho_loss = torch.nn.functional.mse_loss(sim, eye)
                elif 'l1' in slot_ortho_reg_mode:
                    eye = torch.eye(t*k).expand_as(sim).to(sim.device)
                    slot_ortho_loss = torch.nn.functional.l1_loss(sim, eye)
                elif 'ce' in slot_ortho_reg_mode:
                    sim = sim.reshape(bs*k, k)
                    sim = torch.abs(sim)
                    sim_label = torch.arange(k).repeat(bs).long().to(sim.device)  # [0,1,...,k-1,0,1,...,k-1,...]
                    slot_ortho_loss = torch.nn.functional.cross_entropy(sim, sim_label)
                else:
                    raise Exception(f'Un-implemented slot_ortho_reg_mode {slot_ortho_reg_mode}.')

                loss = loss + self.config['args'].slot_ortho_reg_coeff * slot_ortho_loss

                self.epoch_log['scaler']['Tag'].append('loss/slot_ortho_loss')
                self.epoch_log['scaler']['Idx'].append(self.epoch)
                self.epoch_log['scaler']['Value'].append(slot_ortho_loss.item())

                collections['slot_ortho_loss'] = slot_ortho_loss.item()

            # alpha = model.prompt.slot_attn_alpha    # [3]
            # for alpha_idx in range(len(alpha)):
            #     self.epoch_log['scaler']['Tag'].append(f'alpha/{alpha_idx}')
            #     self.epoch_log['scaler']['Idx'].append(self.epoch)
            #     self.epoch_log['scaler']['Value'].append(alpha[alpha_idx].item())
            # loss = 1/alpha[0]**2 * recon_loss
            # loss = loss + 1/alpha[1]**2 * self.mk_coeff * mk_loss
            # loss = loss + 1/alpha[2]**2 * self.slot_vsI_coeff * slot_sim_mse
            # loss = loss + torch.sum(torch.log(alpha+1))

        masked_out = out[:, :self.valid_out_dim]       # [bs, 30]
        bs, n_cls = masked_out.shape

        # ce with heuristic
        logits = masked_out.clone()
        logits[:, :self.last_valid_out_dim] = -float('inf')
        # logits[:, :self.last_valid_out_dim] = logits[:, :self.last_valid_out_dim].detach().clone()
        # dw_cls = self.dw_k[-1 * torch.ones(cat_labels.size()).long()]
        # objs = (self.criterion_fn(logits, cat_labels.long()) * dw_cls).view(bs, pop)  # [bs, n_prompt]

        if 'prompt' in optimizer_target or 'head' in optimizer_target:
            ce_loss = self.criterion_fn(logits, targets.long()).mean()
            loss = loss + ce_loss

            if self.debug_mode:
                print(f'ce_loss: {ce_loss.item()}')

            self.epoch_log['scaler']['Tag'].append('loss/ce_loss')
            self.epoch_log['scaler']['Idx'].append(self.epoch)
            self.epoch_log['scaler']['Value'].append(ce_loss.item())

            collections['ce_loss'] = ce_loss.item()

            cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
            # selection_ortho_loss
            if self.config['args'].use_selection_slot_similar_reg:
                bs, t, k, e, pp = selections.shape   # [bs, t1, k10, e5, pp30]
                batched_selections = selections.reshape(bs, t*k, e*pp)
                bs, t, k, h = slots.shape  # [bs, t1, k30, h128]
                batched_slots = slots.reshape(bs, t*k, h)
                selection_sim = cos(batched_selections.unsqueeze(1), batched_selections.unsqueeze(2))  # [bs, k, k]
                slot_sim = cos(batched_slots.unsqueeze(1), batched_slots.unsqueeze(2))  # [bs, k, k]
                selection_slot_similar_reg_mode = self.config['args'].selection_slot_similar_reg_mode
                if selection_slot_similar_reg_mode == 'l1':
                    selection_ortho_loss = F.l1_loss(selection_sim, slot_sim)
                elif selection_slot_similar_reg_mode == 'l2':
                    selection_ortho_loss = F.mse_loss(selection_sim, slot_sim)
                else:
                    raise Exception(f'Un-implemented selection_slot_similar_reg_mode: '
                                    f'{selection_slot_similar_reg_mode}')

                loss = loss + self.config['args'].selection_slot_similar_reg_coeff * selection_ortho_loss

                self.epoch_log['scaler']['Tag'].append('loss/selection_ortho_loss')
                self.epoch_log['scaler']['Idx'].append(self.epoch)
                self.epoch_log['scaler']['Value'].append(selection_ortho_loss.item())

                collections['selection_ortho_loss'] = selection_ortho_loss.item()

            # prompt-concept alignment loss
            if self.config['args'].use_prompt_concept_alignment_reg:
                bs, t, n, k = attn.shape        # [bs, t1, n196, k30]
                attn = attn.clone().permute(0, 1, 3, 2).reshape(bs, t*k, n)
                bs, channel, height, weight = inputs.shape  # [bs, 3, H, W] [100, 3, 224, 224]

                # random select some inputs to reduce cuda memory cost
                n_samples = 1
                indexs = np.random.permutation(range(bs))[:n_samples]
                select_inputs = inputs[indexs]          # [n_samples, 3, H, W]
                select_targets = targets[indexs]        # [n_samples]
                select_prompts = prompts[indexs]        # [n_samples, tk, e, p, d]
                select_attn = attn[indexs]              # [n_samples, tk, n]

                # expand inputs to match k slots
                k_expand_inputs = select_inputs.repeat_interleave(t*k, dim=0)  # [n_samples*t*k, 3, H, W]
                # k_expand_targets = select_targets.repeat_interleave(t * k)     # [n_samples*tk]
                # expand prompts to match k slots
                bs, t, k, e, p, d = prompts.shape
                k_expand_prompts = select_prompts.reshape(n_samples * t * k, e, p, d)    # [n_samples*t*k, e, p, d]

                # process mask
                with torch.no_grad():
                    # to 0-1 mask
                    select_attn[select_attn >= 0.5] = 1
                    select_attn[select_attn < 0.5] = 0

                    grid_size = 14
                    assert (grid_size * grid_size == n
                            ), f'attn {attn.shape} and grid_size {grid_size} not match. can not put attn on input.'
                    expand_size = height // grid_size
                    select_attn = select_attn.reshape(n_samples*t*k, grid_size, grid_size)

                    # do dilation with 1 neighbor on patch-level
                    kernel_size = 3  # dilate one patch neighbor
                    dilate = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=int(kernel_size / 2))  # 3, 1, 1
                    select_attn = dilate(select_attn)

                    # expand attn to input size
                    masks = select_attn.repeat_interleave(
                        expand_size, dim=1).repeat_interleave(expand_size, dim=2)  # [n_samples*t*k, height, weight]

                    # apply masks
                    k_expand_inputs = torch.einsum('bchw,bhw->bchw', k_expand_inputs, masks)

                # expend masked inputs, targets, prompts accordingly before forward
                k_expand_inputs = k_expand_inputs.reshape(
                    n_samples * t * k, 1, channel, height, weight
                ).repeat_interleave(t*k, dim=1).reshape(
                    n_samples*t*k * t*k, channel, height, weight
                )  # [n_samples*t*k * t*k, 3, H, W]   [000111222]
                # k_expand_targets = k_expand_targets.repeat_interleave(t * k)  # [n_samples*tk*tk]
                k_expand_prompts = k_expand_prompts.reshape(
                    n_samples, 1, t * k, e, p, d
                ).repeat_interleave(t*k, dim=1).reshape(
                    n_samples * t * k * t*k, e, p, d
                )   # [n_samples*t*k * t*k, e, p, d]  [012012012]

                _, k_expand_features = self.model(
                    k_expand_inputs, q=k_expand_prompts,
                    pen=True, train=True,
                    forward_last=False,         # False for only get features if use Aux classifier
                    debug_mode=self.debug_mode)
                # features: [n_samples, 768]

                # check one concept, distance to other prompts, then average.
                emb_dim = k_expand_features.shape[-1]
                k_expand_features = k_expand_features.view(n_samples, t*k, t*k, emb_dim)  # [k(concept), k(prompt)]
                anchor = torch.stack([
                    k_expand_features[:, ki, ki] for ki in range(t*k)
                ], dim=1)       # [n_samples, t*k, emb_dim]

                # cosine sim
                cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                anchor = anchor.reshape(n_samples, t*k, 1, emb_dim)   # same for prompt axis.
                similarities = cos(k_expand_features, anchor)   # [n_samples, k, k]
                avg_similarities = similarities.mean()
                prompt_concept_alignment_loss = avg_similarities

                # last = model.prompt.prompt_concept_alignment_classifier        # [c100, h128]
                # k_expand_logits = last(k_expand_features)    # use aux classifier

                # k_expand_logits = k_expand_logits[:, :self.valid_out_dim]       # [n_samples*k*K, 30]
                # k_expand_logits[:, :self.last_valid_out_dim] = -float('inf')
                # k_expand_logits = k_expand_logits.reshape(n_samples, t*k, t*k, self.valid_out_dim)
                # # positive ce loss and negative entropy loss
                # positive_sample_logits, negative_sample_logits = [], []
                # for kx in range(t*k):
                #     negative_sample_collection = []
                #     for ky in range(t*k):
                #         logits = k_expand_logits[:, kx, ky]     # [n_samples, 30]
                #         if kx == ky:
                #             positive_sample_logits.append(logits)
                #         else:
                #             negative_sample_collection.append(logits)
                #     negative_sample_collection = torch.stack(negative_sample_collection, dim=1)  # [n_samples,k,30]
                #     negative_sample_logits.append(negative_sample_collection)
                # positive_sample_logits = torch.stack(
                #     positive_sample_logits, dim=1
                # ).reshape(n_samples * t * k, self.valid_out_dim)
                # negative_sample_logits = torch.stack(
                #     negative_sample_logits, dim=1
                # ).reshape(n_samples * t * k * t * (k-1), self.valid_out_dim)
                # positive_loss = self.criterion_fn(
                #     positive_sample_logits, k_expand_targets.long()).mean()
                # probs = F.softmax(negative_sample_logits, dim=1)
                # negative_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                # prompt_concept_alignment_loss = positive_loss - 1 * negative_loss

                loss = loss + self.config['args'].prompt_concept_alignment_reg_coeff * prompt_concept_alignment_loss

                if self.debug_mode:
                    print(f'prompt_concept_alignment_loss: {prompt_concept_alignment_loss.item()}')

                self.epoch_log['scaler']['Tag'].append('loss/prompt_concept_alignment_loss')
                self.epoch_log['scaler']['Idx'].append(self.epoch)
                self.epoch_log['scaler']['Value'].append(prompt_concept_alignment_loss.item())

                collections['prompt_concept_alignment_loss'] = prompt_concept_alignment_loss.item()

            # onehot loss
            if self.config['args'].use_selection_onehot_reg:       # and self.epoch >= 5:
                bs, t, k, e, pp = selections.shape   # [bs, t1, k10, e5, pp30]
                batched_selections = selections.reshape(bs*t*k*e, pp)
                with torch.no_grad():
                    onehot_selections = torch.argmax(batched_selections, dim=-1)
                    onehot_selections = F.one_hot(onehot_selections, num_classes=pp)

                if self.config['args'].selection_onehot_reg_mode == 'l1':
                    onehot_loss = F.l1_loss(batched_selections, onehot_selections)
                else:
                    onehot_loss = F.mse_loss(batched_selections, onehot_selections)

                onehot_coeff = self.config['args'].selection_onehot_reg_coeff
                loss = loss + onehot_coeff * onehot_loss

                self.epoch_log['scaler']['Tag'].append('loss/selection_onehot_loss')
                self.epoch_log['scaler']['Idx'].append(self.epoch)
                self.epoch_log['scaler']['Value'].append(onehot_loss.item())

                collections['selection_onehot_loss'] = onehot_loss.item()

            # if self.epoch >= self.epochs - 10:       # left 10 epochs for reg
            if self.config['args'].use_weight_reg:
                # regularization loss on slot2prompt mapping
                # selection: ['weights', 'response']
                reg_mode = self.config['args'].weight_reg_mode
                if self.t > 0 and reg_mode == 'weights':
                    # s2p_loss = torch.stack([
                    #     F.kl_div(torch.log(F.softmax(v.flatten(), dim=0)),
                    #              F.softmax(self.s2p_state_dict[k].flatten(), dim=0), reduction='batchmean')
                    #     for k, v in model.prompt.s2p.named_parameters()
                    # ])
                    s2p_state_dict = copy.deepcopy(self.s2p_copy.state_dict())
                    s2p_loss = torch.stack([
                        torch.norm(v - s2p_state_dict[k], p=2)
                        for k, v in model.prompt.s2p.named_parameters()
                    ])
                    s2p_loss = torch.mean(s2p_loss)
                elif self.t > 0 and reg_mode == 'response':
                    bs, t, k, d = slots.shape
                    _slots = slots.reshape(bs, t * k, d)
                    if len(self.cls_stats) > 0:
                        # align slots with proto and sim over n_cls
                        n_old_cls = len(self.cls_stats)
                        proto = torch.zeros(n_old_cls, *_slots.shape[-2:]).to(slots.device)  # [n_cls, k5, d128]
                        for label in self.cls_stats.keys():
                            proto[label] = self.cls_stats[label]['slots'].detach().clone()
                        proto_ = proto.unsqueeze(0)  # [1, n_cls, k5, d128]
                        proto_ = proto_.unsqueeze(2)  # [1, n_cls, 1, k5, d128]
                        slots_ = slots.unsqueeze(1)  # [bs, 1, k5, d128]
                        slots_ = slots_.unsqueeze(3)  # [bs, 1, k5, 1, d128]
                        # print('proto_', proto_.shape)
                        # print('slots_', slots_.shape)
                        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                        sim = cos(slots_, proto_)  # [bs, n_cls, k, k]  each slot
                        # print('sim', sim.shape)
                        cost = 1 - sim
                        batch_cost, index = hungarian_algorithm(
                            cost.reshape(bs * n_old_cls, t * k, t * k))  # [bs*n_cls, k], [bs*n_cls, 2, k]
                        batch_cost = batch_cost.reshape(bs, n_old_cls, t * k)
                        # indexes = index.reshape(bs, n_old_cls, 2, t * k)
                        batch_sim = 1 - batch_cost
                        aligned_sim = torch.sum(batch_sim, dim=-1)  # [bs, n_cls] sim over aligned-slot

                        aligned_sim = aligned_sim ** p

                        # cal beta, sum over n_cls and softmax over batch
                        beta = torch.softmax(torch.sum(aligned_sim, dim=1), dim=0)
                    else:
                        beta = torch.ones(bs).to(out.device) / bs     # [bs]: as mean [1/bs, 1/bs,...]

                    if self.debug_mode:
                        print(f'beta: {beta}')

                    # kl on response without target logits
                    # out [bs, 1, n_cls]
                    # remove target logits
                    selected_out = masked_out.reshape(bs, n_cls)

                    if self.debug_mode:
                        print(f'selected_out {selected_out.shape}: {selected_out[0]}')

                    mask = torch.arange(n_cls).expand(bs, n_cls).to(targets.device) != targets.unsqueeze(1)

                    if self.debug_mode:
                        print(f'targets {targets.shape}: {targets[0]}')
                        print(f'mask {mask.shape}: {mask[0]}')

                    selected_out = selected_out[mask].view(bs, n_cls - 1)

                    if self.debug_mode:
                        print(f'selected_out {selected_out.shape}: {selected_out[0]}')

                    # y_one_hot = F.one_hot(targets, num_classes=n_cls)
                    # mask = 1 - y_one_hot
                    # selected_out = torch.stack([selected_out[bi][mask[bi]] for bi in range(bs)])      # [bs, n_cls-1]
                    selected_out = selected_out[:, self.last_valid_out_dim:]    # [bs, 10-1class]

                    if self.debug_mode:
                        print(f'selected_out {selected_out.shape}: {selected_out[0]}')

                    selected_out = torch.softmax(selected_out / tau, dim=1)

                    if self.debug_mode:
                        print(f'selected_out after softmax {selected_out.shape}: {selected_out[0]}')

                    # obtain old response
                    with torch.no_grad():
                        slots = slots.reshape(bs, t*k, d)
                        old_prompts = model.prompt.slot2prompt(slots, self.s2p_copy)    # [bs, e, p, d]
                        _, _, old_out = self.obtain_mo_matrix(
                            None, prompts=old_prompts.unsqueeze(1),     # [bs, 1, e, p, d]
                            train=True,
                            samples=inputs,
                            labels=targets,
                            group_by_labels=False,
                            return_features=True,
                        )
                        old_out = old_out[:, :, :self.valid_out_dim]
                        # ce with heuristic
                        old_out[:, :, :self.last_valid_out_dim] = -float('inf')
                        selected_old_out = old_out.reshape(bs, n_cls)

                        if self.debug_mode:
                            print(f'selected_old_out {selected_old_out.shape}: {selected_old_out[0]}')

                        selected_old_out = selected_old_out[mask].view(bs, n_cls - 1)

                        if self.debug_mode:
                            print(f'selected_old_out {selected_old_out.shape}: {selected_old_out[0]}')

                        selected_old_out = selected_old_out[:, self.last_valid_out_dim:]    # [bs, 10-1class]

                        if self.debug_mode:
                            print(f'selected_old_out {selected_old_out.shape}: {selected_old_out[0]}')

                        selected_old_out = torch.softmax(selected_old_out / 3, dim=1)

                        if self.debug_mode:
                            print(f'selected_old_out after softmax {selected_old_out.shape}: {selected_old_out[0]}')

                    # kl on selected_out and selected_old_out
                    s2p_loss = (tau ** 2) * F.kl_div(torch.log(selected_out), selected_old_out, reduction='none')

                    if self.debug_mode:
                        print(f's2p_loss {s2p_loss.shape}: {s2p_loss[0]}')

                    s2p_loss = beta * s2p_loss.sum(dim=-1)
                    s2p_loss = s2p_loss.sum()

                    if self.debug_mode:
                        print(f's2p_loss {s2p_loss.shape}: {s2p_loss}')

                weight_coeff = self.config['args'].weight_reg_coeff
                loss = loss + weight_coeff * s2p_loss

                self.epoch_log['scaler']['Tag'].append('loss/s2p_loss')
                self.epoch_log['scaler']['Idx'].append(self.epoch)
                self.epoch_log['scaler']['Value'].append(s2p_loss.item())

                collections['s2p_loss'] = s2p_loss.item()

                # if self.debug_mode:
                #     print(f'grad after: {next(model.prompt.s2p[0].parameters()).grad[0,0]}')

            ext_logits, ext_targets = [], []
            ext_slots, ext_slot_weights, ext_w_slots = [], [], []
            if self.config['args'].use_old_samples_for_reg and self.t > 0:
                # append some old samples
                old_inputs, old_targets = self.aux.sampling()

                if self.config['args'].use_old_samples_for_reg_no_grad:
                    with torch.no_grad():
                        res = self.forward(
                            old_inputs, old_targets,
                            train=True, learn_slots=False, prompt_phase=prompt_phase)
                else:
                    res = self.forward(old_inputs, old_targets,
                                       train=True, learn_slots=False, prompt_phase=prompt_phase)
                old_slots = res['slots']
                old_slot_weights = res['slot_weights']
                old_w_slots = res['w_slots']
                old_logits = res['logits'][:,:self.valid_out_dim]
                ext_logits.append(old_logits)
                ext_targets.append(old_targets)
                ext_slots.append(old_slots)
                ext_slot_weights.append(old_slot_weights)
                ext_w_slots.append(old_w_slots)

            ext_logits.append(logits)
            ext_targets.append(targets)
            ext_slots.append(slots)
            ext_w_slots.append(w_slots)
            ext_slot_weights.append(slot_weights)
            ext_logits = torch.cat(ext_logits, dim=0)
            ext_targets = torch.cat(ext_targets, dim=0)
            ext_slots = torch.cat(ext_slots, dim=0)
            ext_slot_weights = torch.cat(ext_slot_weights, dim=0)
            ext_w_slots = torch.cat(ext_w_slots, dim=0)

            # cheating reg on logits
            if self.concept_weight:
                concept_similar_reg = self._concept_similar_reg(None, ext_logits, ext_targets)
                self.epoch_log['scaler']['Tag'].append('loss/concept_similar_reg')
                self.epoch_log['scaler']['Idx'].append(self.epoch)
                self.epoch_log['scaler']['Value'].append(concept_similar_reg.item())

                collections['concept_similar_reg'] = concept_similar_reg.item()

                # cal current_coeff
                coeff = self.config['concept_similar_reg_coeff']
                sen = self.config['concept_similar_reg_coeff_sensitivity']
                last_coeff = ((10 / self.n_cls) ** sen) * coeff
                if self.config['args'].dynamic_concept_similar_reg_coeff:
                    current_coeff = last_coeff * (self.epoch+1) / self.epochs
                else:
                    current_coeff = last_coeff
                self.epoch_log['scaler']['Tag'].append(f'coeff/concept_similar_reg')
                self.epoch_log['scaler']['Idx'].append(self.epoch)
                self.epoch_log['scaler']['Value'].append(current_coeff)

                loss = loss + current_coeff * concept_similar_reg

            if self.config['args'].use_slot_logit_similar_reg:
                # cal current_coeff
                coeff = self.config['args'].slot_logit_similar_reg_coeff
                sen = self.config['args'].slot_logit_similar_reg_coeff_sensitivity
                current_coeff = ((10 / self.n_cls) ** sen) * coeff
                self.epoch_log['scaler']['Tag'].append(f'coeff/slot_logit_similar_reg_coeff/t{self.t}')
                self.epoch_log['scaler']['Idx'].append(self.epoch)
                self.epoch_log['scaler']['Value'].append(current_coeff)

                # if 'slot' in optimizer_target:
                # slot_logit_similar_reg = self._slot_logit_similar_reg(
                #     ext_slots, ext_slot_weights, ext_w_slots, ext_logits)
                # else:
                slot_logit_similar_reg = self._slot_logit_similar_reg(
                    ext_slots.detach(), ext_slot_weights.detach(), ext_w_slots.detach(), ext_logits)
                # detach slots and weights
                self.epoch_log['scaler']['Tag'].append('loss/slot_logit_similar_reg')
                self.epoch_log['scaler']['Idx'].append(self.epoch)
                self.epoch_log['scaler']['Value'].append(slot_logit_similar_reg.item())

                collections['slot_logit_similar_reg'] = slot_logit_similar_reg.item()

                loss = loss + current_coeff * slot_logit_similar_reg

        loss.backward()

        # step
        self.optimizer.step()

        # logits is used to cal train acc, so use masked out (-inf for old) to show local acc
        return loss.detach(), logits, collections

    def _intra_consistency_reg(self, slots, slot_weights, w_slots, targets):
        bs, t, k, h = slots.shape  # [bs, t1, k30, h128]
        img_slots = slots.reshape(bs, t * k, h)

        mode = self.config['args'].intra_consistency_reg_mode
        if 'learn' in mode:
            # learned slot selection
            bs, t, k = slot_weights.shape
            img_weights = slot_weights.reshape(bs, t * k)           # .detach()
            weighted_slots = torch.einsum('bkh,bk->bh', img_slots, img_weights)
        elif 'map' in mode:
            bs, t, h = w_slots.shape
            weighted_slots = w_slots.reshape(bs, t*h)
        elif 'cross' in mode:
            # cross attn
            weights = self.cross_attn(img_slots)        # [bs, k]
            weighted_slots = torch.einsum('bkh,bk->bh', img_slots, weights)
        else:
            raise Exception(f'Un-implemented intra_consistency_reg_mode: {mode}')

        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        if 'kl' in mode:
            targets_1hot = F.one_hot(targets).float()
            label_sim = cos(targets_1hot.unsqueeze(1), targets_1hot.unsqueeze(0))      # [bs, bs]
            label_sim = label_sim / label_sim.sum(dim=-1, keepdim=True)    # l1-norm
            if 'dot' in mode:
                sim = weighted_slots @ weighted_slots.t()     # [b,b]
                intra_consistency_loss = cross_entropy_with_soft_labels(sim, label_sim)
            else:
                sim = cos(weighted_slots.unsqueeze(0), weighted_slots.unsqueeze(1))     # [b,b]
                intra_consistency_loss = cross_entropy_with_soft_labels(sim, label_sim)
                # sim = cos(weighted_slots.unsqueeze(1), weighted_slots.unsqueeze(0)
                #           ) * self.config['args'].slot_logit_similar_reg_slot_temp  # [bs, bs]
                # sim = (sim - sim.min(dim=-1, keepdim=True)[0]) / (
                #         sim.max(dim=-1, keepdim=True)[0] - sim.min(dim=-1, keepdim=True)[0] + 1e-10)
                # # minmax over row to make them positive
                # sim = sim / sim.sum(dim=-1, keepdim=True)  # l1-norm
                #
                # intra_consistency_loss = cross_entropy_with_soft_labels(sim, label_sim, normalized=True)

        else:
            if 'l1' in mode:
                dist = nn.PairwiseDistance(p=1)  # l1-distance
            elif 'l2' in mode:
                dist = nn.PairwiseDistance(p=2)
            else:
                raise Exception(f'Un-implemented intra_consistency_reg_mode: {mode}')

            # group by labels
            intra_consistency_loss = []
            labels = torch.unique(targets)
            for label in labels:
                selected_slots = weighted_slots[targets == label]
                if 'cos' in mode:
                    sim = cos(selected_slots.unsqueeze(0), selected_slots.unsqueeze(1))     # [b,b]
                    intra_consistency_loss.append(dist(sim, torch.ones_like(sim)).flatten())
                else:
                    intra_consistency_loss.append(dist(selected_slots.unsqueeze(0), selected_slots.unsqueeze(1)).flatten())
            intra_consistency_loss = torch.concat(intra_consistency_loss)
            if intra_consistency_loss.shape[0] == bs:       # all samples are different labels, loss is all 0
                intra_consistency_loss = intra_consistency_loss.sum()
            else:
                intra_consistency_loss = intra_consistency_loss.sum() / (intra_consistency_loss.shape[0] - bs)
            # all values include bs self-dist cal (which is 0)

        # # # find a positive sample for each sample (if only has one sample in this batch, use itself)
        # # posi_slots = []
        # # for sid in range(bs):
        # #     target = targets[sid]
        # #     selected_idxs = torch.where(targets == target)[0]
        # #     selected_idx = selected_idxs[torch.randperm(selected_idxs.size(0))][0]
        # #     posi_slot = weighted_slots[selected_idx]
        # #     posi_slots.append(posi_slot)
        # # posi_slots = torch.stack(posi_slots)  # [bs, h]
        #
        # intra_consistency_loss = dist(weighted_slots, posi_slots).mean()

        return intra_consistency_loss

    def _concept_similar_reg(self, features, logits, targets):
        """Cheating on concept-aware to decrease distance between two imgs that share the same concept"""
        # features: [bs 768]; logits with full range: [bs, 100]; targets: [bs]

        # preprocess logits
        logits = logits[:, self.last_valid_out_dim:self.valid_out_dim]      # [bs, 10]

        # collect concepts
        # self.label_concepts: [100, 2]
        concepts_batch = self.label_concepts[targets.cpu().numpy()]   # [bs, 2]
        num_concepts = self.num_concepts
        concepts = self.train_dataset.process_concepts(
            torch.from_numpy(concepts_batch).long(), num_concepts).to(logits.device)
        # [bs, n_concepts]

        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        concept_sim = cos(concepts.unsqueeze(1), concepts.unsqueeze(0)) * 1  # [bs, bs]
        concept_sim = concept_sim / concept_sim.sum(dim=-1, keepdim=True)    # l1-norm
        if 'cos' in self.config['args'].concept_similar_reg_mode:
            logit_sim = cos(logits.unsqueeze(1), logits.unsqueeze(0)) * 5
        else:       # dot
            logit_sim = torch.matmul(logits, logits.t()) * (
                    self.config['args'].concept_similar_reg_temp * (logits.shape[-1] ** -0.5))

        if 'l2' in self.config['args'].concept_similar_reg_mode:
            loss = F.mse_loss(torch.sigmoid(logit_sim), concept_sim)
        elif 'l1' in self.config['args'].concept_similar_reg_mode:
            loss = F.l1_loss(torch.sigmoid(logit_sim), concept_sim)
        else:       # kl
            loss = cross_entropy_with_soft_labels(logit_sim, concept_sim)

        return loss

    def _slot_logit_similar_reg(self, slots, weights, w_slots, logits):
        """contrastive on weighted slots and logits
        """
        if 'map' in self.config['args'].slot_logit_similar_reg_mode:
            bs, t, h = w_slots.shape
            weighted_slots = w_slots.reshape(bs, t*h)
        else:
            bs, t, k, d = slots.shape
            batched_slots = slots.reshape(bs, t * k, d)
            bs, t, k = weights.shape
            batched_weights = weights.reshape(bs, t * k)
            weighted_slots = torch.einsum('bkd,bk->bd', batched_slots, batched_weights)

        # preprocess logits
        logits = logits[:, self.last_valid_out_dim:self.valid_out_dim]      # [bs, 10]

        if 'cos' in self.config['args'].slot_logit_similar_reg_mode:
            cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
            slot_sim = cos(weighted_slots.unsqueeze(1), weighted_slots.unsqueeze(0)
                           ) * self.config['args'].slot_logit_similar_reg_slot_temp  # [bs, bs]
            normed_slot_sim = (slot_sim - slot_sim.min(dim=-1, keepdim=True)[0]) / (
                    slot_sim.max(dim=-1, keepdim=True)[0] - slot_sim.min(dim=-1, keepdim=True)[0] + 1e-10)
            # minmax over row to make them positive
            normed_slot_sim = normed_slot_sim / normed_slot_sim.sum(dim=-1, keepdim=True)  # l1-norm
            # logit_sim = cos(logits.unsqueeze(1), logits.unsqueeze(0)) * 5
            logit_sim = torch.matmul(logits, logits.t()) * (
                    self.config['args'].slot_logit_similar_reg_temp * (logits.shape[-1] ** -0.5))
        else:
            slot_sim = torch.matmul(weighted_slots, weighted_slots.t()) * (
                    self.config['args'].slot_logit_similar_reg_slot_temp * weighted_slots.shape[-1] ** -0.5)
            normed_slot_sim = F.softmax(slot_sim, dim=-1)
            logit_sim = torch.matmul(logits, logits.t()) * (
                    self.config['args'].slot_logit_similar_reg_temp * (logits.shape[-1] ** -0.5))

        if 'l2' in self.config['args'].slot_logit_similar_reg_mode:
            loss = F.mse_loss(logit_sim, slot_sim)
        elif 'l1' in self.config['args'].slot_logit_similar_reg_mode:
            loss = F.l1_loss(logit_sim, slot_sim)
        # elif 'ce' in self.config['args'].slot_logit_similar_reg_mode:
        #     distances = F.l1_loss(logit_sim, slot_sim, reduction='none')    # [bs, bs]
        #     F.cross_entropy(distances, torch.range(0, distances.shape[0] - 1).long().to(distances.device))
        else:
            loss = cross_entropy_with_soft_labels(logit_sim, normed_slot_sim)

        return loss

    def cross_attn(self, slots):
        # cross-attn and sum for all bs*k slots
        bs, k, h = slots.shape
        q = nn.functional.normalize(slots, dim=-1)
        # q = nn.functional.normalize(slots.detach(), dim=-1)
        # q = q.reshape(bs, -1, h)  # [bs, k, h]
        weights = torch.sum(q.unsqueeze(2) * q.reshape(1, 1, bs * k, h), dim=-1)
        # [bs, k, bs*k] cosine sim matrix
        weights = torch.mean(weights, dim=-1)  # [bs, k]
        weights = weights * self.config['args'].slot_cross_attn_temp  # cross_attn_temp
        weights = torch.softmax(weights, dim=-1)  # [b, k]; sum over k == 1
        return weights

    def forward_mk(self, slots, K):
        # slot_attn_class_key
        s = self.last_valid_out_dim
        f = self.valid_out_dim
        K = K[:f]
        if self.t > 0:
            K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
        n_K = nn.functional.normalize(K, dim=1)

        weights = self.cross_attn(slots)
        q = torch.einsum('bkh,bk->bh', slots, weights)

        mk_logit = torch.einsum('bh,ch->bc', q, n_K)  # wei-sum over h -> cos sim
        return mk_logit, weights

    def update_model_f4m(self, inputs, targets, match_pool=False, learn_slots=False):
        self.optimizer.zero_grad()
        try:
            model = self.model.module
        except:
            model = self.model
        FPS = self.FPS

        # obtain slots
        # if match_pool:
        #     # use pool's slots
        #     with torch.no_grad():
        #         prompts = model.obtain_q(inputs)      # [bs, t, k20, e12, p8, d768]
        #     prompts = model.prompt.match_pool(prompts)
        # else:
        #     prompts = model.obtain_q(inputs, all=False)      # [bs, 1, k20, e12, p8, d768]
        #     # all=False: only use new slots
        q = model.obtain_q(inputs, learn_slots=learn_slots)      # [bs, 1, k20, e12, p8, d768]
        prompts, selection, ws, w_slots, slots, attn, recon_loss = q

        if learn_slots:
            # only update slot attn
            loss = torch.mean(torch.stack(recon_loss))       # list [1\T]

            if self.debug_mode:
                print(f'slot recon loss: {loss.item()}')

            self.epoch_log['scaler']['Tag'].append('loss/slot_recon_loss')
            self.epoch_log['scaler']['Idx'].append(self.epoch)
            self.epoch_log['scaler']['Value'].append(loss.item())

            loss.backward()

            # step
            self.optimizer.step()

            logits = None
            return loss.detach(), logits, None

        else:
            bs, t, k, e, p, d = prompts.shape
            prompts = prompts.reshape(bs, t*k, e, p, d)
            # prompts = prompts[:, -1]

            # forward all slots without grad to obtain loss matrix used to select minimal-5 for grads.
            # for l in self.e_layers:
            # with torch.no_grad():
            mo_matrix, features, out = self.obtain_mo_matrix(
                None, prompts=prompts,
                train=True,
                samples=inputs,
                labels=targets,
                group_by_labels=False,
                return_features=True,
            )  # [bs, 20]

            if self.debug_mode:
                print(f'mo_matrix {mo_matrix.shape}')

            # # select n_opt_slots minimal slots for each img
            # indexs = torch.sort(mo_matrix, dim=1)[1][:, :self.n_opt_slots]
            # indexs = torch.stack([indexs for _ in range(slots.shape[-1])], dim=-1)      # [bs, 5, 64]
            # slots = torch.gather(slots, dim=1, index=indexs)  # [bs, 5, h64]
            # # slots = torch.stack([slots[idx, indexs[idx]] for idx in range(indexs.shape[0])])  # [bs, 5, h64]
            #
            # mo_matrix, features = self.obtain_mo_matrix(
            #     None, slots=slots,
            #     train=True,
            #     samples=inputs,
            #     labels=targets,
            #     group_by_labels=False,
            #     return_features=True,
            # )  # [bs, 5]
            # # features: [bs, 5, 768]

            # # sort new slots according to loss
            # mo_matrix = torch.cat([
            #     mo_matrix[:, :(t-1) * k],                           # old slots
            #     torch.sort(mo_matrix[:, (t-1) * k:], dim=1)[0]      # sorted new slots
            # ], dim=1)
            #
            # n_opt_slots = (t-1) * k + self.n_opt_slots      # self.n_opt_slots; mo_matrix.shape[-1]
            # # if self.epoch < self.config['schedule'][-1] / 3:        # 10 for 30 epochs
            # #     n_opt_slots = mo_matrix.shape[-1]
            # # else:
            # #     n_opt_slots = self.n_opt_slots
            # mo_matrix = mo_matrix[:, :n_opt_slots]   # [bs, 5]

            sorted_mo_matrix, indexes = torch.sort(mo_matrix, dim=1)      # [bs, 10]

            '''log'''
            for obj_idx in range(sorted_mo_matrix.shape[0]):
                for pop_idx in range(sorted_mo_matrix.shape[1]):
                    self.epoch_log['mo']['Tag'].append('loss')
                    self.epoch_log['mo']['Pop_id'].append(pop_idx)
                    self.epoch_log['mo']['Obj_id'].append(obj_idx)
                    self.epoch_log['mo']['Epoch_id'].append(self.epoch)
                    self.epoch_log['mo']['Inner_id'].append(0)
                    self.epoch_log['mo']['Value'].append(sorted_mo_matrix[obj_idx, pop_idx].item())

            positive = sorted_mo_matrix[:, :self.n_opt_slots]       # [bs, 2]

            loss = torch.mean(positive, dim=1)        # [bs]
            loss = torch.mean(loss)

            # loss.backward(retain_graph=True)

            if self.debug_mode:
                print(f'loss: {loss.item()}')

            self.epoch_log['scaler']['Tag'].append('loss/mo_loss')
            self.epoch_log['scaler']['Idx'].append(self.epoch)
            self.epoch_log['scaler']['Value'].append(loss.item())

            # # generate expert labels
            # # indexes[0]:                   tensor([6, 4, 5, 2, 7, 9, 1, 3, 8, 0], device='cuda:0')
            # # selected_positive_indexes[0]: tensor([6, 4], device='cuda:0')
            # # selected_negative_indexes[0]: tensor([8, 0], device='cuda:0')
            # # selected_indexes[0]:          tensor([0, 4, 6, 8], device='cuda:0')
            # # label_indexes[0]:             tensor([3, 1, 0, 2], device='cuda:0')
            # # labels[0]:                    tensor([0, 1, 1, 0], device='cuda:0')
            # assert 3*self.n_opt_slots <= t*k    # n_opt_slots needs to be smaller than n_slots/3  1:2 samples
            # selected_positive_indexes = indexes[:, :self.n_opt_slots]
            # selected_negative_indexes = indexes[:, -2*self.n_opt_slots:]
            # selected_indexes, label_indexes = torch.sort(
            #     torch.cat([selected_positive_indexes, selected_negative_indexes], dim=1), dim=1)
            # labels = torch.zeros_like(selected_indexes).long()      # [bs, 6]
            # labels[:, :self.n_opt_slots] = 1
            # labels = torch.stack([labels[bid, label_indexes[bid]] for bid in range(bs)])    # [bs, 6]
            # # labels = torch.zeros_like(selected_indexes).long()      # [bs, 6]
            # # labels.scatter_(1, indexes[:, :self.n_opt_slots], 1)        # expert: 1; not expert: 0 [bs*10]
            # labels = labels.flatten()       # [bs*6]
            #
            # # forward expert classifier
            # expert_predictor = model.prompt.expert_predictor[-1]
            # # exp_out = expert_predictor(features.reshape(-1, features.size(-1)))     # [bs*10, 2]
            # exp_out = out.reshape(bs*t*k, out.size(-1)).detach()      # [bs*10, 100]
            # exp_out = expert_predictor(exp_out[:, self.last_valid_out_dim:self.valid_out_dim])      # [bs*10, 2]
            # selected_out = exp_out.reshape(bs, t*k, 2)
            # selected_out = torch.stack([selected_out[bid, selected_indexes[bid]] for bid in range(bs)])
            # selected_out = selected_out.reshape(-1, 2)       # [bs*6, 2]
            #
            # exp_loss = self.criterion_fn(selected_out, labels.long())     # [bs*6, 2]
            # exp_loss = torch.mean(exp_loss, dim=-1)     # [bs*6]
            # # # balance importance for 1: 2:8 -> 1:4; [0, 1,..., 1] -> [1, 4,..., 4]
            # # weights = labels*9+1
            # # # weights = labels*((labels.shape[0]-torch.sum(labels))/torch.sum(labels)-1)+1
            # # exp_loss = exp_loss * weights
            # exp_loss = torch.mean(exp_loss)
            #
            # if self.debug_mode:
            #     print(f'expert loss: {exp_loss.item()}')
            #
            # self.epoch_log['scaler']['Tag'].append('loss/expert_loss')
            # self.epoch_log['scaler']['Idx'].append(self.epoch)
            # self.epoch_log['scaler']['Value'].append(exp_loss.item())
            #
            # # if expert_pred_flag:
            # #     total_loss = exp_loss
            # # else:
            # total_loss = loss + exp_loss
            # # exp_loss = torch.zeros(1).mean()

            # exp_out = torch.argmax(exp_out, dim=-1).reshape(bs, t, k)[:, 0]         # [bs, 10]
            # # # if no experts for 1 img, then use all experts
            # # exp_out[torch.sum(exp_out, dim=-1) == 0] = 1     # all 0 means no expert, use all experts: all 1
            # # change all 0 to -1
            # exp_out[exp_out == 0] = -1

            out = out[:, :, :self.valid_out_dim]
            bs, n_slots, n_cls = out.shape
            # ce with heuristic
            out[:, :, :self.last_valid_out_dim] = -float('inf')

            # regularization loss: entropy
            selected_negative_indexes = indexes[:, self.n_opt_slots:]
            negative = torch.stack([
                out[bid, selected_negative_indexes[bid], self.last_valid_out_dim:] for bid in range(bs)])

            entropy_loss = calc_entropy(negative)
            self.epoch_log['scaler']['Tag'].append('loss/entropy_loss')
            self.epoch_log['scaler']['Idx'].append(self.epoch)
            self.epoch_log['scaler']['Value'].append(entropy_loss.item())

            total_loss = loss + entropy_loss
            total_loss.backward()

            # voting [bs, 20, 100] -> [bs, 100]
            out = torch.argmax(out, dim=-1)  # [bs, 20]
            # out = (out * exp_out).long()     # apply expert mask, thus only experts' selection is positive.
            logits = torch.zeros(bs, n_cls).to(out.device)
            for cls_id in range(n_cls):
                logits[:, cls_id] = torch.sum(out == cls_id, dim=-1)

            # step
            self.optimizer.step()

            return loss.detach(), logits, entropy_loss.detach()

    def return_front(self, mo_matrix, idx=-1):
        """Return the specific front from mo_matrix: [obj, pop]"""
        n_obj, n_pop = mo_matrix.shape
        mo_obj = mo_matrix.detach().cpu().numpy()

        # non-dom sorting to create multiple fronts
        subfront_indices = fastNonDominatedSort(mo_obj)
        number_of_fronts = np.max(subfront_indices) + 1  # +1 because of 0 indexing

        if idx >= number_of_fronts or idx == -1:
            idx = number_of_fronts - 1      # return the last front (dominated ones)

        return mo_matrix[:, subfront_indices == idx]

    def obtain_mo_matrix(self, hard_l=None, prompts=None, addition=None,
                         check=False, train=True,
                         samples=None, labels=None,
                         return_nui_labels=False,
                         group_by_labels=False,
                         return_features=False,
                         ):
        """Return mo_matrix: Torch tensor [obj, pop]
        Obj: samples; Pop: prompts
        If addition is not None: [num, h...], append and return [obj, pop+num]
        If train is False, do not use -inf to mask past tasks' logits
        """
        def sampling(n_obj, min_samples=10):
            """Sample and check whether number of list of samples contains at least min_samples
            is larger than n_obj, and return cat-ed samples and labels.
            :param n_obj: number of column that need to satisfied.
            :param min_samples:

            :return cat-ed samples and labels.
            """
            _dict = {}
            while len([_dict[_l] for _l in _dict.keys() if _dict[_l].shape[0] >= min_samples]) < n_obj:
                ss, ls, _ = self.aux.sampling(self.num_aux_sampling, sort=True)  # [100, 3, 224, 224]
                for label in sorted(set(ls)):
                    if label in _dict:
                        _dict[label] = torch.cat((_dict[label], ss[ls == label]))
                    else:
                        _dict[label] = ss[ls == label]

            _samples, _labels = [], []
            for label, ss in _dict.items():
                if ss.shape[0] >= min_samples:
                    _samples.append(ss[:min_samples])
                    _labels.append(torch.fill_(torch.empty(min_samples), label))

            _samples = torch.stack(_samples)        # [10, 10, 3, 224, 224] [n_cls, min_samples, *img_size]
            _labels = torch.stack(_labels)          # [10, 10]  [n_cls, min_samples]
            index = np.random.choice(len(_samples), n_obj, replace=False)

            return _samples[index].view(n_obj*min_samples, *_samples.shape[-3:]), _labels[index].view(n_obj*min_samples)

        if samples is None:
            '''sampling by aux'''
            num_sample_per_obj = 3
            samples, labels = sampling(2, min_samples=num_sample_per_obj)
            # dead while when n_obj > n_class_per_task

            if self.gpu:
                samples = samples.cuda()
                labels = labels.cuda()

        n_samples = samples.shape[0]
        nui_labels = torch.unique(labels)
        # check if any label has only 1 sample
        if check:
            check = True
            flag = []
            for label in nui_labels:
                if len(labels[labels == label]) < 2:
                    check = False
                    flag.append(label.item())
            if not check:
                print(f'mo: label ({flag}) has/have only 1 sample')
                return None

        ncc_losses = None
        features = None
        out = None
        if hard_l is None or hard_l in self.e_layers:       # None for all layer to use specific prompt

            if prompts is None:
                print('prompts are NONE.')
                # obtain slots
                try:
                    model = self.model.module
                except:
                    model = self.model
                q = model.obtain_q(samples)        # [bs, t, k20, e12, p8, d768]
                prompts, selection, ws, w_slots, slots, attn, recon_loss = q
                bs, t, k, e, p, d = prompts.shape
                prompts = prompts.reshape(bs, t*k, e, p, d)

            if addition is not None:
                print('addition is not NONE.')
                # add to slots
                addition = torch.stack([addition for _ in range(n_samples)])        # [bs, num, ...]
                prompts = torch.cat([prompts, addition], dim=1)        # [bs, k20+num, ...]

            len_p = prompts.shape[1]
            stack_p = prompts.reshape(n_samples * len_p, *prompts.shape[2:])  # [bs*k, ...]

            # rearrange samples labels and hard_obj_idx
            stack_samples = torch.stack([samples for _ in range(len_p)],
                                        dim=1).reshape(n_samples*len_p, *samples.shape[1:])
            # [bs * n_slots, 3, 224, 224]

            if self.debug_mode:
                print(n_samples, len_p, '\nmo samples:', stack_samples.shape, 'prompts:', stack_p.shape)

            # pen: penultimate features; train: same forward as batch training.
            out, features = self.model(
                stack_samples, q=stack_p,
                pen=True, train=train,
                # register_blk=hard_l,
                debug_mode=self.debug_mode)
            # features: [bs*n_slots, 768]
            # out is logits: [bs*n_slots, 100]

            ## pair-wise sim loss
            out = out.reshape(n_samples, len_p, out.shape[-1])   # [bs, n_slots, 100]
            features = features.reshape(n_samples, len_p, features.shape[-1])   # [bs, n_slots, 768]

            ncc_losses = self.obtain_loss(out, labels, mode='last', train=train, group_by_labels=group_by_labels)
            # [bs, n_slots]

        if return_nui_labels:
            return ncc_losses, nui_labels
        if return_features:
            return ncc_losses, features, out

        return ncc_losses

    def obtain_loss(self, out, labels, mode='last', train=False, group_by_labels=True):
        nui_labels = torch.unique(labels)
        if mode == 'last':
            # use ce loss
            # out: [bs, n_prompt, 100]
            logits = out
            # broadcast labels
            bs, pop, _ = logits.shape
            logits = logits.view(-1, logits.size(-1))       # [bs*n_prompt, 100]
            cat_labels = torch.stack([labels for _ in range(pop)], dim=1).view(-1)  # [bs*n_prompt]

            logits = logits[:, :self.valid_out_dim]
            # ce with heuristic
            if train:
                logits[:, :self.last_valid_out_dim] = -float('inf')
            # logits[:, :self.last_valid_out_dim] = logits[:, :self.last_valid_out_dim].detach().clone()
            # dw_cls = self.dw_k[-1 * torch.ones(cat_labels.size()).long()]
            # objs = (self.criterion_fn(logits, cat_labels.long()) * dw_cls).view(bs, pop)  # [bs, n_prompt]
            objs = (self.criterion_fn(logits, cat_labels.long())).view(bs, pop)  # [bs, n_prompt]
            if group_by_labels:
                '''group objectives [n_samples, n_prompt] -> [n_label, n_prompt]'''
                ncc_losses = torch.stack([torch.mean(objs[labels == label], dim=0) for label in nui_labels])
            else:
                ncc_losses = objs
        elif mode in ['cos', 'dot', 'l2']:
            # use prototype loss
            # out: [bs, 768]
            features = out
            # group according to labels
            ncc_losses = []
            grouped_features = []     # each group of samples can have different numbers, can not stack
            anchors = []
            for label in nui_labels:
                label_features = features[labels == label]          # [n_img, n_prompt+, 768]
                grouped_features.append(label_features)

                # obtain anchor
                anchor = torch.mean(label_features, dim=0)       # [n_prompt+, 768]
                anchors.append(anchor)

            anchors = torch.stack(anchors, dim=0)       # [n_labels, n_prompt+, 768]
            anchors = anchors.unsqueeze(0)              # [1, n_label, n_prompt+, 768]

            # loss
            for idx, label in enumerate(nui_labels):
                label_features = grouped_features[idx]          # [n_img, n_prompt+, 768]
                label_features = label_features.unsqueeze(1)    # [n_img, 1, n_prompt+, 768]

                if mode == 'cos':
                    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                    sim = cos(anchors, label_features)  # [n_img, n_label, n_prompt+]
                    # [-1+1]
                    # sim[idx] - sum(sim[others/{idx}])
                    posi_sim = 1-sim[:, idx]
                    nega_sim = torch.cat([sim[:, :idx], sim[:, idx+1:]], dim=1)
                    nega_sim[nega_sim < 0] = 0      # max{nega_sim, 0}
                    sim = posi_sim + torch.sum(nega_sim, dim=1)     # [n_img, n_prompt+]
                    # option: or use ce loss on sim -> sim as logits
                    ncc_losses.append(torch.mean(sim, dim=0))     # [n_prompt+]
                # elif dis == 'dot':
                #     dist = - (anchors * label_features).sum(-1)
                #     ncc_losses.append(torch.mean(sim, dim=0))     # [n_prompt+]
                # elif dis == 'l2':
                #     d = nn.PairwiseDistance(p=2)
                #     dist = d(anchors, label_features)     # [0, +inf]
                #     ncc_losses.append(torch.mean(dist, dim=0))     # [n_prompt+]
                else:
                    raise Exception('Unknown distance {}'.format(mode))

                ncc_losses = torch.stack(ncc_losses)        # [n_label(obj), n_prompt (new, old, add_concepts)]

        else:
            raise Exception(f'not implemented mode: {mode}')

        return ncc_losses

    def process_concepts(self, concepts, num_prompts):
        # # from [bs, 1, 2] -> [bs, num_prompts]  multi-hot float
        # concepts = concepts[:, 0]       # [bs, 2]
        # concept_labels = F.one_hot(concepts, num_prompts)
        # concept_labels = torch.sum(concept_labels, dim=1).float()

        # # from [bs, 1, num_prompts] -> [bs, num_prompts]  multi-hot float
        concept_labels = concepts[:, 0]       # [bs, 21]

        return concept_labels

    def validation(self, dataloader, model=None, task_in=None, task_metric='acc', verbal=True, task_global=False,
                   during_train=False, slot_recon_loss=False, use_slot_statistics=False):
        """during_train=True -> local val acc (mask on current logits) no use"""
        # return 0
        # pass task to forward if task-awareness
        if model is None:
            model = self.model
        try:
            model_single = model.module
        except:
            model_single = model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        # mk_acc = AverageMeter()
        # collect_top_k = (1, 3, 5, 7, 10)
        # mk_task_acc = AverageMeter(top_k=collect_top_k)
        recon_losses = AverageMeter()
        silhouette_scores = AverageMeter()
        max_attns = AverageMeter()
        batch_timer.tic()

        # logit_task_mask_top_k = self.config['logit_task_mask_top_k']

        orig_mode = model.training
        model.eval()

        # # load statistics for evaluating
        # self.load_statistics()

        mode = 'head'
        if use_slot_statistics:         # but shape: [n_cls, 128]
            assert len(self.cls_stats) != 0
            mode = 'slot'
            cls_stats = torch.stack([self.cls_stats[label]['slots'] for label in range(len(self.cls_stats))])
            # [n_cls, 768]    # will raise exception if label is not from 0->n_cls-1
            cls_stats = nn.functional.normalize(cls_stats, dim=1)

        elif self.config['args'].use_feature_statistics and len(self.cls_stats) != 0:
            mode = 'feature'
            cls_stats = torch.stack([self.cls_stats[label]['features'] for label in range(len(self.cls_stats))])
            # [n_cls, 768]    # will raise exception if label is not from 0->n_cls-1
            cls_stats = nn.functional.normalize(cls_stats, dim=1)

        label_task_map = np.zeros(self.config['num_classes'])
        for task_id, task in enumerate(self.tasks):
            for ta in task:
                label_task_map[ta] = task_id
        for i, sample in enumerate(dataloader):
            concepts = None
            if len(sample) == 3:
                (input, target, task) = sample
            else:   # contain concepts
                (input, target, concepts, task) = sample

            if self.debug_mode and i == 0:
                print(
                    f'eval batch{i}: \nlen: {len(target)} target:{(target.min(), target.max())} '
                    f'task:{(task.min(), task.max())}')
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            #         if concepts is not None:
            #             concepts = concepts.cuda()  # [bs, 1, 21]
            # if concepts is not None:
            #     concepts = self.process_concepts(concepts, self.num_prompts)        # [bs, 21]
            with torch.no_grad():
                if task_in is None:
                    # output = model.forward(input, task_id=task[0].item())[:, :self.valid_out_dim]
                    # output = model.forward(input,
                    #                        concepts=concepts if self.use_concept_labels_as_aqk else None
                    #                        )[:, :self.valid_out_dim]

                    res = self.forward(input, target)
                    prompts = res['prompts']
                    selections = res['selections']
                    slot_weights = res['slot_weights']
                    slots = res['slots']
                    attn = res['attn']
                    recon_loss = res['recon_loss']
                    out = res['logits']
                    features = res['features']

                    # q = model_single.obtain_q(input)  # [bs, t, k20, e12, p8, d768]
                    # prompts, selection, ws, w_slots, slots, attn, recon_loss = q
                    bs, t, k, h = slots.shape  # [bs, t1, k30, h128]
                    # assert t == 1

                    # # slot_attn_class_key
                    # K = model.prompt.slot_attn_class_key  # [c100, h128]
                    # s = self.last_valid_out_dim
                    # mk_logit, mk_weights = self.forward_mk(slots.reshape(bs, t*k, h), K)
                    #
                    # if self.debug_mode and i == 0:
                    #     print(f'mk_logit: {mk_logit.shape} {mk_logit[0]}')
                    #
                    # # mk_class_acc
                    # mk_acc = accumulate_acc(mk_logit, target, task, mk_acc, topk=(self.top_k,))
                    #
                    # task_ids = torch.empty_like(target)     # [bs]
                    # _, mk_pred = mk_logit.topk(collect_top_k[-1], 1, True, True)    # [bs, topk]
                    # mk_task_pred = torch.empty_like(mk_pred)    # [bs, topk]
                    # for idx in range(bs):
                    #     for idxx in range(collect_top_k[-1]):
                    #         mk_task_pred[idx, idxx] = label_task_map[mk_pred[idx, idxx].item()]
                    #     task_ids[idx] = label_task_map[target[idx].item()]
                    #
                    # if self.debug_mode and i == 0:
                    #     print(f'task_ids: {task_ids.shape} {task_ids[0]}'
                    #           f'mk_task_pred: {mk_task_pred.shape} {mk_task_pred[0]}')
                    #
                    # mk_task_pred = mk_task_pred.t()
                    # correct = mk_task_pred.eq(task_ids.reshape(1, -1).expand_as(mk_task_pred))
                    # # correct: BOOL [topk, bs]
                    #
                    # if self.debug_mode and i == 0:
                    #     print(f'correct: {correct.shape} {correct[0]}')
                    #
                    # res = []
                    # for k in collect_top_k:
                    #     # correct_k = correct[:k].reshape(-1).float().sum().item()
                    #     correct_k = correct[:k].float().sum(dim=0)
                    #     correct_k = (correct_k > 0).sum().item()        # >0 -> True for multiple correct
                    #     res.append(correct_k * 100.0 / bs)
                    # mk_task_acc.update(res, bs)

                    if slot_recon_loss:
                        recon_loss = torch.mean(torch.stack(recon_loss))  # list [1\T]

                        # # collect slot mean
                        # slot_mean = torch.mean(torch.abs(slots).reshape(bs, -1), dim=1)        # [bs]
                        # recon_loss = torch.mean(slot_mean)      # record slot mean

                        # collect slot group average silhouette_score
                        if i == 0: # only do it for the first batch
                            # slots [bs, t, k, h128]
                            collect_slots = slots.reshape(-1, slots.shape[-1])  # [bs*t*k10, h128]

                            X = collect_slots.detach().cpu().numpy()

                            # normalization
                            scaler = MinMaxScaler()
                            X_normalized = scaler.fit_transform(X)

                            # Initialize the clusterer with n_clusters value and a random generator
                            # seed of 10 for reproducibility.
                            n_clusters = 30
                            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                            cluster_labels = clusterer.fit_predict(X_normalized)  # np [k10]
                            silhouette_avg = silhouette_score(X_normalized, cluster_labels)
                            silhouette_scores.update(silhouette_avg, collect_slots.shape[0])

                        # collect attn statistics, average over patch: max over slot
                        # attn [bs, t, n196, k10]
                        attn = attn.reshape(-1, attn.shape[-1])       # [bs*t*n196, k10]
                        max_attn = torch.max(attn, dim=-1)[0]
                        max_attn = torch.mean(max_attn)
                        max_attns.update(max_attn.item(), bs)

                        recon_losses.update(recon_loss.item(), bs)
                        continue

                    # selection metric
                    # selections: [bs, 1, k10, e12, pp30]

                    # # forward all prompts
                    # bs, t, k, e, p, d = prompts.shape
                    # prompts = prompts.reshape(bs, t*k, e, p, d)
                    # prompts = torch.sum(prompts, dim=1, keepdim=True)     # sum over k [bs, 1, e, p, d]
                    # _, features, out = self.obtain_mo_matrix(
                    #     None, prompts=prompts,
                    #     train=False,
                    #     samples=input,
                    #     labels=target,
                    #     group_by_labels=False,
                    #     return_features=True,
                    # )
                    # if len(self.cls_stats) == 0:
                    #     out = out[:, :, :self.valid_out_dim]
                    #     # out: [bs, t*20, self.valid_out_dim] if during_train: [-inf,..., value] else: [value,..., value]
                    #
                    #     if self.debug_mode and i == 0:
                    #         print(f'out: {out[0, 0]}')
                    #         print(f'valid_out_dim: {self.valid_out_dim}')
                    #
                    #     output = out.reshape(bs, -1)  # [bs, 1, n_cls] -> [bs, n_cls]
                    # else:
                    #     features = features.reshape(bs, -1)     # [bs, 1, 768] -> [bs, 768]
                    #     features = nn.functional.normalize(features, dim=1)
                    #     output = torch.einsum('bd,cd->bc', features, cls_stats)

                    # use slot to cal cos sim
                    if mode == 'slot':
                        # w_slots = res['w_slots']  # [bs, t, 128]
                        # w_slots = w_slots.reshape(bs, -1)
                        bs, t, k, d = slots.shape
                        slots = slots.reshape(bs, t * k, d)  # [bs, k10, d128]
                        bs, t, k = slot_weights.shape
                        slot_weights = slot_weights.reshape(bs, t*k)
                        w_slots = torch.einsum('bkd,bk->bd', slots, slot_weights)
                        w_slots = nn.functional.normalize(w_slots, dim=1)
                        output = torch.einsum('bd,cd->bc', w_slots, cls_stats)
                    elif mode == 'feature':
                        features = features.reshape(bs, -1)     # [bs, 1, 768] -> [bs, 768]
                        features = nn.functional.normalize(features, dim=1)
                        output = torch.einsum('bd,cd->bc', features, cls_stats)
                    else:
                        out = out[:, :self.valid_out_dim]
                        # out: [bs, t*20, self.valid_out_dim] if during_train: [-inf,..., value] else: [value,..., value]

                        if self.debug_mode and i == 0:
                            print(f'out: {out[0]}')
                            print(f'valid_out_dim: {self.valid_out_dim}')

                        output = out.reshape(bs, -1)  # [bs, 1, n_cls] -> [bs, n_cls]

                    # if logit_task_mask_top_k > 0:
                    #     # apply logit_task_mask_top_k on output
                    #     logit = torch.zeros_like(output) * -float('inf')
                    #     mk_task_pred = mk_task_pred.t()     # [bs, topk]
                    #
                    #     for sample_id in range(bs):
                    #         for task_id in range(logit_task_mask_top_k):
                    #             ta = mk_task_pred[sample_id, task_id].item()
                    #             logit[sample_id, self.tasks[ta]] = output[sample_id, self.tasks[ta]]
                    #
                    # else:
                    logit = output

                    if self.debug_mode and i == 0:
                        print(f'masked logit: {logit.shape} {logit[0]}')

                    acc = accumulate_acc(logit, target, task, acc, topk=(self.top_k,))
                else:
                    mask = target >= task_in[0]
                    mask_ind = mask.nonzero().view(-1)
                    input, target = input[mask_ind], target[mask_ind]

                    mask = target <= task_in[-1]
                    mask_ind = mask.nonzero().view(-1)
                    input, target = input[mask_ind], target[mask_ind]

                    # if i == 0:
                    #     print('DEBUG: '
                    #         f'eval batch{i}: \nlen: {len(target)} target:{(target.min(), target.max())} '
                    #         f'task:{(task.min(), task.max())}')

                    if len(target) > 1:
                        res = self.forward(input, target)
                        prompts = res['prompts']
                        selections = res['selections']
                        slot_weights = res['slot_weights']
                        slots = res['slots']
                        attn = res['attn']
                        recon_loss = res['recon_loss']
                        out = res['logits']
                        features = res['features']
                        bs, t, k, h = slots.shape  # [bs, t1, k10, h128]

                        # q = model_single.obtain_q(input)  # [bs, t, k20, e12, p8, d768]
                        # prompts, selection, ws, w_slots, slots, attn, recon_loss = q
                        # # bs, t, k, e, p, d = prompts.shape
                        # # prompts = prompts.reshape(bs, t * k, e, p, d)
                        # # # slots = model_single.prompt.match_pool(slots)
                        # bs, t, k, h = slots.shape  # [bs, t1, k30, h128]
                        # assert t == 1

                        # # slot_attn_class_key
                        # K = model.prompt.slot_attn_class_key  # [c100, h128]
                        # s = self.last_valid_out_dim
                        # mk_logit, mk_weights = self.forward_mk(slots.reshape(bs, t*k, h), K)
                        #
                        # if self.debug_mode and i == 0:
                        #     print(f'mk_logit: {mk_logit.shape} {mk_logit[0]}')
                        #
                        # # mk_class_acc
                        # if not task_global:
                        #     mk_logit = mk_logit[:, task_in]
                        # if task_global:
                        #     mk_acc = accumulate_acc(mk_logit, target, task, mk_acc, topk=(self.top_k,))
                        # else:
                        #     mk_acc = accumulate_acc(mk_logit, target - task_in[0], task, mk_acc, topk=(self.top_k,))
                        #
                        # task_ids = torch.empty_like(target)     # [bs]
                        # _, mk_pred = mk_logit.topk(collect_top_k[-1], 1, True, True)    # [bs, topk]
                        # mk_task_pred = torch.empty_like(mk_pred)    # [bs, topk]
                        # for idx in range(bs):
                        #     for idxx in range(collect_top_k[-1]):
                        #         mk_task_pred[idx, idxx] = label_task_map[mk_pred[idx, idxx].item()]
                        #     task_ids[idx] = label_task_map[target[idx].item()]
                        # mk_task_pred = mk_task_pred.t()
                        # correct = mk_task_pred.eq(task_ids.reshape(1, -1).expand_as(mk_task_pred))
                        # res = []
                        # for k in collect_top_k:
                        #     # correct_k = correct[:k].reshape(-1).float().sum().item()
                        #     correct_k = correct[:k].float().sum(dim=0)
                        #     correct_k = (correct_k > 0).sum().item()
                        #     res.append(correct_k * 100.0 / bs)
                        # mk_task_acc.update(res, bs)

                        if slot_recon_loss:
                            recon_loss = torch.mean(torch.stack(recon_loss))  # list [1\T]
                            recon_losses.update(recon_loss.item(), bs)
                            continue

                        # # forward all prompts
                        # bs, t, k, e, p, d = prompts.shape
                        #
                        # # if i == 0:
                        # #     print('DEBUG: '
                        # #           f'prompts: {prompts.shape}')
                        #
                        # prompts = prompts.reshape(bs, t * k, e, p, d)
                        # prompts = torch.sum(prompts, dim=1, keepdim=True)     # sum over k [bs, 1, e, p, d]
                        # _, features, out = self.obtain_mo_matrix(
                        #     None, prompts=prompts,
                        #     train=False,
                        #     samples=input,
                        #     labels=target,
                        #     group_by_labels=False,
                        #     return_features=True,
                        # )
                        # if len(self.cls_stats) == 0:
                        #     out = out[:, :, :self.valid_out_dim]
                        #     # out: [bs, t*20, self.valid_out_dim] if during_train: [-inf,..., value] else: [value,..., value]
                        #
                        #     output = out.reshape(bs, -1)  # [bs, 1, n_cls] -> [bs, n_cls]
                        # else:
                        #     features = features.reshape(bs, -1)  # [bs, 1, 768] -> [bs, 768]
                        #     features = nn.functional.normalize(features, dim=1)
                        #     output = torch.einsum('bd,cd->bc', features, cls_stats)
                        #
                        #     # if not task_global:
                        #     #     output = output[:, task_in]

                        # use slot to cal cos sim
                        if mode == 'slot':
                            # w_slots = res['w_slots']  # [bs, t, 128]
                            # w_slots = w_slots.reshape(bs, -1)
                            bs, t, k, d = slots.shape
                            slots = slots.reshape(bs, t * k, d)  # [bs, k10, d128]
                            bs, t, k = slot_weights.shape
                            slot_weights = slot_weights.reshape(bs, t*k)
                            w_slots = torch.einsum('bkd,bk->bd', slots, slot_weights)
                            w_slots = nn.functional.normalize(w_slots, dim=1)
                            output = torch.einsum('bd,cd->bc', w_slots, cls_stats)
                            output = output[:, :self.valid_out_dim]
                        elif mode == 'feature':
                            features = features.reshape(bs, -1)  # [bs, 1, 768] -> [bs, 768]
                            features = nn.functional.normalize(features, dim=1)
                            output = torch.einsum('bd,cd->bc', features, cls_stats)
                            output = output[:, :self.valid_out_dim]
                        else:
                            out = out[:, :self.valid_out_dim]
                            # out: [bs, t*20, self.valid_out_dim] if during_train: [-inf,..., value] else: [value,..., value]

                            output = out.reshape(bs, -1)  # [bs, 1, n_cls] -> [bs, n_cls]

                            # if not task_global:
                            #     output = output[:, task_in]

                        if task_global:
                            # output = model.forward(input, task_id=task[0].item())[:, :self.valid_out_dim]
                            output = output[:, :self.valid_out_dim]
                            acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                        else:
                            # output = model.forward(input, task_id=task[0].item())[:, task_in]
                            output = output[:, task_in]

                            # print(f'DEBUG: task_in: {task_in}\n output: {output.shape}\n target: {target}')

                            acc = accumulate_acc(output, target - task_in[0], task, acc, topk=(self.top_k,))

        model.train(orig_mode)

        # mk_task_acc.avg = np.array([np.round(v, 3) for v in mk_task_acc.avg])
        if slot_recon_loss:
            if verbal:
                self.log(' * Val Recon Loss {recon_losses.avg:.3e}, '
                         'Silhouette Score {silhouette_score.avg:.3f}, '
                         'Avg Max Attn {max_attn.avg:.3f}, '
                         'Total time {time:.2f}'
                         .format(recon_losses=recon_losses, silhouette_score=silhouette_scores,
                                 max_attn=max_attns,
                                 # mk_acc=mk_acc, mk_task_acc=mk_task_acc,
                                 time=batch_timer.toc(),    # collect_top_k=collect_top_k
                                 ))
                # 'MK Acc {mk_acc.avg:.3f}, '
                # 'MK Top{collect_top_k} Task Acc {mk_task_acc.avg}'
            return max_attns.avg
            # return recon_losses.avg
        else:
            if verbal:
                local=''
                if task_in is not None:
                    local=' local'
                self.log(' * {mode}: Val Acc {acc.avg:.3f}, '
                         'Total time {time:.2f}{local}'
                         .format(mode=mode,
                                 acc=acc,
                                 # mk_acc=mk_acc, mk_task_acc=mk_task_acc,
                                 time=batch_timer.toc(),    # collect_top_k=collect_top_k
                                 local=local,
                                 ))
                # 'MK Acc {mk_acc.avg:.3f}, '
                # 'MK Top{collect_top_k} Task Acc {mk_task_acc.avg}'
            return acc.avg

    def collect_slot_pool(self, train_loader, train_dataset, model=None):
        task_id = train_dataset.t
        if model is None:
            model = self.model
        batch_timer = Timer()
        batch_timer.tic()
        orig_mode = model.training
        model.eval()
        batch_timer.tic()

        slots_collection = []
        with torch.no_grad():
            for i, sample in enumerate(train_loader):
                if i*train_loader.batch_size > 100: # collect enough samples
                    break
                concepts = None
                if hasattr(train_dataset, "return_concepts") and train_dataset.return_concepts:
                    x, y, concepts, task = sample
                else:
                    x, y, task = sample

                # send data to gpu
                if self.gpu:
                    x = x.cuda()
                    y = y.cuda()

                # get slots
                q = model.obtain_q(x, learn_slots=False)
                _, _, _, slots, _, _, _ = q  # [bs, 1, k5, d128]
                bs, t, k, d = slots.shape
                slots = slots.reshape(bs, t * k, d)  # [bs, k5, d128]
                slots_collection.append(slots)
            slots_collection = torch.cat(slots_collection, dim=0).reshape(-1, d)    # [100*k, d128]
            slots_collection = slots_collection.cpu().numpy()

            n_clusters = 50
            slot_centers, _, _ = k_means(slots_collection, n_clusters=n_clusters, random_state=0)

        model.train(orig_mode)

        self.log(' * Collect slot_centers: Total time {time:.2f}'
                 .format(time=batch_timer.toc()))

        # save statistics
        stats_path = os.path.join(self.config['log_dir'], 'temp', f'slot_centers_{task_id}.pkl')
        print('=> Saving statistics to:', stats_path)
        with open(stats_path, 'wb') as f:
            pickle.dump(slot_centers, f)
        print('=> Save Done')

    def collect_statistics(self, train_loader, train_dataset, model=None, refresh=False):
        if model is None:
            model = self.model

        # refresh cls_stats for different tasks
        if refresh:
            self.cls_stats = {}
            self.cls_stats_n = {}

        batch_timer = Timer()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
        batch_timer.tic()
        for i, sample in enumerate(train_loader):
            concepts = None
            if hasattr(train_dataset, "return_concepts") and train_dataset.return_concepts:
                x, y, concepts, task = sample
            else:
                x, y, task = sample

            # send data to gpu
            if self.gpu:
                x = x.cuda()
                y = y.cuda()
                # if concepts is not None:
                #     concepts = concepts.cuda()      # [bs, 224, 224]
                # task = task.cuda()

            with torch.no_grad():

                res = self.forward(x, y, model=model)
                prompts = res['prompts']
                selections = res['selections']
                slot_weights = res['slot_weights']
                slots = res['slots']
                attn = res['attn']
                recon_loss = res['recon_loss']
                out = res['logits']
                features = res['features']

                # # get slots
                # q = model.obtain_q(x, learn_slots=False)
                # prompts, _, _, _, _, _, _ = q  # [bs, 1, k5, d128]
                # bs, t, k, e, p, d = prompts.shape
                # prompts = prompts.reshape(bs, t*k, e, p, d)
                #
                # sum_prompts = torch.sum(prompts, dim=1)  # [bs, e, p, d]
                #
                # # pen: penultimate features; train: same forward as batch training.
                # _, features = model(
                #     x, q=sum_prompts,
                #     forward_last=False,
                #     pen=True, train=False)
                # # features: [bs, 768]

                # label-wise collecting prototypes
                labels = torch.unique(y)
                for label in labels:
                    label = label.item()

                    # collect slots with this label
                    labeled_features = features[y == label]  # [n_img, 768]
                    n_img = labeled_features.size(0)
                    avg_features = torch.mean(labeled_features, dim=0)

                    if label in self.cls_stats.keys() and 'features' in self.cls_stats[label].keys():
                        prev_features = self.cls_stats[label]['features']
                        prev_n_img = self.cls_stats_n[label]
                        avg_features = (prev_features * prev_n_img + avg_features * n_img) / (prev_n_img + n_img)
                        n_img = prev_n_img + n_img
                    elif label not in self.cls_stats.keys():
                        self.cls_stats[label] = {}

                    self.cls_stats[label]['features'] = avg_features
                    self.cls_stats_n[label] = n_img

        model.train(orig_mode)

        self.log(' * Collect statistics: Total time {time:.2f}'
                 .format(time=batch_timer.toc()))

    def collect_slot_statistics(self, train_loader, train_dataset, model=None, save=False, refresh=False):
        """
        Collect slot statistics for each label. weighted sum slot [n_cls, h128]
        Using slot_weights [bs, k10]
        """
        task_id = train_dataset.t
        seed = self.config['seed']

        if self.load_statistics(t=task_id, seed=seed, name='slot_stats'):
            return

        if model is None:
            model = self.model

        try:
            prompt = model.module.prompt
        except:
            prompt = model.prompt

        # refresh cls_stats for different tasks
        if refresh:
            self.cls_stats = {}
            self.cls_stats_n = {}

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        batch_timer.tic()

        buffer = dict()
        orig_mode = model.training
        model.eval()
        batch_timer.tic()
        for i, sample in enumerate(train_loader):
            concepts = None
            if hasattr(train_dataset, "return_concepts") and train_dataset.return_concepts:
                x, y, concepts, task = sample
            else:
                x, y, task = sample

            # send data to gpu
            if self.gpu:
                x = x.cuda()
                y = y.cuda()
                # if concepts is not None:
                #     concepts = concepts.cuda()      # [bs, 224, 224]
                # task = task.cuda()

            with torch.no_grad():
                # get slots
                res = self.forward(x, y)
                prompts = res['prompts']
                selections = res['selections']
                slot_weights = res['slot_weights']
                w_slots = res['w_slots']
                slots = res['slots']
                attn = res['attn']
                recon_loss = res['recon_loss']
                out = res['logits']
                features = res['features']

                # q = model.obtain_q(x, learn_slots=False)
                # _, _, _, slots, _, _, _ = q  # [bs, 1, k10, d128]
                bs, t, k, d = slots.shape
                slots = slots.reshape(bs, t * k, d)  # [bs, k10, d128]
                bs, t, k = slot_weights.shape
                slot_weights = slot_weights.reshape(bs, t*k)

                w_slots = torch.einsum('bkd,bk->bd', slots, slot_weights)

                # bs, t, d = w_slots.shape
                # w_slots = w_slots.reshape(bs, t*d)
                assert t == 1

                # label-wise collecting prototypes
                labels = torch.unique(y)
                for label in labels:
                    label = label.item()

                    # collect slots with this label
                    labeled_slots = w_slots[y == label]  # [n_img, d128]
                    n_img = labeled_slots.size(0)
                    avg_slots = torch.mean(labeled_slots, dim=0)

                    if label in self.cls_stats.keys() and 'slots' in self.cls_stats[label].keys():
                        prev_slots = self.cls_stats[label]['slots']
                        prev_n_img = self.cls_stats_n[label]
                        avg_slots = (prev_slots * prev_n_img + avg_slots * n_img) / (prev_n_img + n_img)
                        n_img = prev_n_img + n_img
                    elif label not in self.cls_stats.keys():
                        self.cls_stats[label] = {}

                    self.cls_stats[label]['slots'] = avg_slots
                    self.cls_stats_n[label] = n_img

                # using buffer to collect a set of samples for mean and variances
                # labels = torch.unique(y)
                # for label in labels:
                #     label = label.item()
                #
                #     # only collect once
                #     if label in self.cls_stats.keys():
                #         continue
                #
                #     # collect slots with this label
                #     labeled_slots = w_slots[y == label]  # [n_img, d128]
                #     if label in buffer.keys():
                #         labeled_slots = torch.cat([labeled_slots, buffer[label]])
                #         del(buffer[label])
                #
                #     # not enough put into buffer
                #     if len(labeled_slots) < 50:      # 50
                #         buffer[label] = labeled_slots
                #         continue
                #
                #     self.cls_stats[label] = torch.mean(labeled_slots, dim=0)  # [d]

        model.train(orig_mode)

        self.log(' * Collect statistics: Len {len}, Total time {time:.2f}'
                 .format(len=len(self.cls_stats), time=batch_timer.toc()))

        if save:
            # save statistics
            stats_path = os.path.join(self.config['log_dir'], 'temp', f'slot_stats_seed{seed}_t{task_id}.pkl')
            print('=> Saving statistics to:', stats_path)
            with open(stats_path, 'wb') as f:
                pickle.dump(self.cls_stats, f)
            print('=> Save Done')

    def collect_slot_statistics_all(self, train_loader, train_dataset, model=None, save=False):
        t = train_dataset.t
        if model is None:
            model = self.model
        # self.cls_stats = {}
        try:
            prompt = model.module.prompt
        except:
            prompt = model.prompt

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        batch_timer.tic()

        buffer = dict()
        orig_mode = model.training
        model.eval()
        batch_timer.tic()
        for i, sample in enumerate(train_loader):
            concepts = None
            if hasattr(train_dataset, "return_concepts") and train_dataset.return_concepts:
                x, y, concepts, task = sample
            else:
                x, y, task = sample

            # send data to gpu
            if self.gpu:
                x = x.cuda()
                y = y.cuda()
                # if concepts is not None:
                #     concepts = concepts.cuda()      # [bs, 224, 224]
                # task = task.cuda()

            with torch.no_grad():
                # get slots
                q = model.obtain_q(x, learn_slots=False)
                _, _, _, slots, _, _, _ = q  # [bs, 1, k5, d128]
                bs, t, k, d = slots.shape
                slots = slots.reshape(bs, t * k, d)  # [bs, k5, d128]

                # label-wise collecting prototypes
                labels = torch.unique(y)
                for label in labels:
                    label = label.item()

                    # only collect once
                    if label in self.cls_stats.keys():
                        continue

                    # collect slots with this label
                    labeled_slots = slots[y == label]  # [n_img, k5, d128]
                    if label in buffer.keys():
                        labeled_slots = torch.cat([labeled_slots, buffer[label]])
                        del(buffer[label])

                    # not enough put into buffer
                    if len(labeled_slots) < 50:      # 50
                        buffer[label] = labeled_slots
                        continue

                    # align slots with 1st slot by hungarian_algorithm
                    # cost matrix -> 1-cossim with 1st slot -> [n_img, k, k]
                    anchor = labeled_slots[:1].unsqueeze(2)  # [1, k5, 1, d128]
                    labeled_slots_ = labeled_slots.unsqueeze(1)  # [n_img, 1, k5, d128]
                    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                    sim = cos(anchor, labeled_slots_)  # [n_img, k, k]
                    cost = 1 - sim
                    _, index = hungarian_algorithm(cost)  # [n_img, 2, k]

                    # apply index
                    slot_indexes = index[:, 1]  # [n_img, k]
                    labeled_slots = torch.stack([labeled_slots[idx, slot_indexes[idx]] for idx in
                                                 range(labeled_slots.shape[0])])  # [n_img, k, d128]

                    self.cls_stats[label] = torch.mean(labeled_slots, dim=0)  # [k, d]

        model.train(orig_mode)

        self.log(' * Collect statistics: Total time {time:.2f}'
                 .format(time=batch_timer.toc()))

        if save:
            # save statistics
            stats_path = os.path.join(self.config['log_dir'], 'temp', f'cls_stats.pkl')
            print('=> Saving statistics to:', stats_path)
            with open(stats_path, 'wb') as f:
                pickle.dump(self.cls_stats, f)
            print('=> Save Done')

    def load_statistics(self, t=0, seed=0, name='cls_stats'):
        stats_path = os.path.join(self.config['log_dir'], 'temp', f'{name}_seed{seed}_t{t}.pkl')

        if os.path.exists(stats_path):
            print('=> Load statistics from:', stats_path)
            with open(stats_path, 'rb') as f:
                self.cls_stats = pickle.load(f)
            return True
        else:
            print('=> Statistics not find.')
            return False

    def predict_mo(self, mo_features):
        """predict based on cls_stats"""
        # # calculate softmax-ed variance
        # mo_logits = torch.softmax(torch.var(mo_features, dim=-1), dim=-1)  # [bs, 21]
        #
        # n_cls = len(self.cls_stats) // 2        # pattern and n_img
        # target = torch.stack([torch.softmax(torch.var(self.cls_stats[idx], dim=-1), dim=-1)
        #                       for idx in range(n_cls)], dim=0)      # [n_cls, 21]
        #
        # logits = -F.kl_div(mo_logits.unsqueeze(1).log(),
        #                    target.unsqueeze(0), reduction='none').mean(-1)    # [bs, n_cls]
        #
        # logits = torch.softmax(logits, dim=-1)      # [bs, n_cls]

        # calculate softmax-ed cos-sim
        norm_features = nn.functional.normalize(mo_features, dim=-1)

        n_cls = len(self.cls_stats) // 2  # pattern and n_img
        target = torch.stack([self.cls_stats[idx]
                              for idx in range(n_cls)], dim=0)  # [n_cls, 21, 768]
        norm_target = nn.functional.normalize(target, dim=-1)

        # cosine sim
        logits = torch.einsum('bpd,cpd->bc', norm_features, norm_target)  # [bs, n_cls]

        return logits


class Auxiliary:
    """
    Provide auxiliary samples for supporting evaluating prompts

    """
    def __init__(self, args, source=None, tasks=None, t=0):
        """source can be a val_dataset"""
        self.args = args
        self.dataset = source
        self.dataset_loader = None
        self.dataset_loader_yield = None
        self.t = t
        self.tasks = tasks
        self.single_class_datasets = {}
        self.single_class_dataset_dataloaders = {}
        self.single_class_dataset_dataloaders_yield = {}
        self.bs = args.batch_size

    def update_source(self, source, t):
        self.dataset = source
        self.t = t
        tasks = self.tasks
        if self.dataset.old_dataset is not None:
            self.dataset_loader = torch.utils.data.DataLoader(
                self.dataset.old_dataset, batch_size=self.bs,
                shuffle=True, drop_last=False, num_workers=self.args.workers
            )
            self.dataset_loader_yield = iter(self.dataset_loader)

        for old_task_id in range(t):
            for class_id in tasks[old_task_id]:
                self.single_class_datasets[class_id] = self.dataset.get_single_class_dataset(
                    class_id, dataset=self.dataset.old_dataset)
                self.single_class_dataset_dataloaders[class_id] = torch.utils.data.DataLoader(
                    self.single_class_datasets[class_id], batch_size=self.bs,
                    shuffle=True, drop_last=False, num_workers=self.args.workers)
                self.single_class_dataset_dataloaders_yield[class_id] = iter(
                    self.single_class_dataset_dataloaders[class_id])

    def update_dataloader(self):
        if self.dataset_loader_yield is not None:
            self.dataset_loader_yield = iter(self.dataset_loader)
        for old_task_id in range(self.t):
            for class_id in self.tasks[old_task_id]:
                if self.single_class_dataset_dataloaders_yield[class_id] is not None:
                    self.single_class_dataset_dataloaders_yield[class_id] = iter(
                        self.single_class_dataset_dataloaders[class_id])

    def sampling(self, split_sampling=False):
        inputs = []
        targets = []
        if split_sampling:
            for old_task_id in range(self.t):
                for class_id in self.tasks[old_task_id]:
                    sample = next(self.single_class_dataset_dataloaders_yield[class_id])

                    if len(sample) == 3:
                        x, y, task = sample
                    else:
                        x, y, c, task = sample
                    # send data to gpu
                    x = x.cuda()
                    y = y.cuda()

                    inputs.append(x)
                    targets.append(y)

        else:
            sample = next(self.dataset_loader_yield)
            if len(sample) == 3:
                x, y, task = sample
            else:
                x, y, c, task = sample
            # send data to gpu
            x = x.cuda()
            y = y.cuda()
            # task is incorrect because it is ori_idx

            inputs.append(x)
            targets.append(y)

        if len(inputs) > 0:
            inputs = torch.cat(inputs, dim=0)
            targets = torch.cat(targets, dim=0)

            # if not split_sampling:
            #     # random select bs samples
            #     selected = np.random.permutation(range(len(inputs)))[:bs]
            #     inputs = inputs[selected]
            #     targets = targets[selected]

        return inputs, targets

    def sampling_old(self, num_samples=100, sort=True):
        """
        :param num_samples:
        :param sort: return sorted samples according to targets. Inactivated if no target provided.
        """
        assert self.source is not None
        indexs = np.arange(len(self.source))
        selected = np.random.choice(indexs, num_samples, replace=True if num_samples > len(indexs) else False)

        imgs = []
        targets = []
        concepts = []
        for idx in selected:
            data = self.source[idx]
            imgs.append(data[0])
            targets.append(data[1])
            if len(data) > 2:
                concepts.append(data[2])
        imgs = torch.stack(imgs)
        targets = np.stack(targets)
        if len(concepts) > 1:
            concepts = torch.stack(concepts)

        if sort:
            sorted_indexs = np.argsort(targets)
            imgs = imgs[sorted_indexs]
            targets = targets[sorted_indexs]
            if len(concepts) > 1:
                concepts = concepts[sorted_indexs]

        if len(concepts) > 1:
            return imgs, torch.from_numpy(targets), concepts
        else:
            return imgs, torch.from_numpy(targets)


def calc_entropy(input_tensor):
    # input_tensor [bs, slots, logits]
    lsm = nn.LogSoftmax(dim=1)      # over slots
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.mean()
    return entropy


def hungarian_algorithm(cost_matrix: Tensor):
    """ Borrowed from https://github.com/JindongJiang/latent-slot-diffusion/blob/master/src/eval/eval_utils.py
    Batch-applies the hungarian algorithm to find a matching that minimizes the overall cost.
    Returns the matching indices as a LongTensor with shape (batch size, 2, min(num objects, num slots)).
    The first column is the row indices (the indices of the true objects) while the second
    column is the column indices (the indices of the slots). The row indices are always
    in ascending order, while the column indices are not necessarily.
    The outputs are on the same device as `cost_matrix` but gradients are detached.
    A small example:
                | 4, 1, 3 |
                | 2, 0, 5 |
                | 3, 2, 2 |
                | 4, 0, 6 |
    would result in selecting elements (1,0), (2,2) and (3,1). Therefore, the row
    indices will be [1,2,3] and the column indices will be [0,2,1].
    Args:
        cost_matrix: Tensor of shape (batch size, num objects, num slots).
    Returns:
        A tuple containing:
            - a Tensor with shape (batch size, min(num objects, num slots)) with the
              costs of the matches.
            - a LongTensor with shape (batch size, 2, min(num objects, num slots))
              containing the indices for the resulting matching.
    """

    # List of tuples of size 2 containing flat arrays
    indices = list(map(linear_sum_assignment, cost_matrix.cpu().detach().numpy()))
    indices = torch.LongTensor(np.array(indices))
    smallest_cost_matrix = torch.stack(
        [
            cost_matrix[i][indices[i, 0], indices[i, 1]]
            for i in range(cost_matrix.shape[0])
        ]
    )
    device = cost_matrix.device
    return smallest_cost_matrix.to(device), indices.to(device)


def cross_entropy_with_soft_labels(logits, soft_targets, normalized=False):
    """
    Calculate the cross-entropy loss for soft labels.

    Args:
        logits: Raw, unnormalized scores output from the model (shape: [batch_size, num_classes]).
        soft_targets: Probability distributions over classes (soft labels) (shape: [batch_size, num_classes]).

    Returns:
        The mean cross-entropy loss with soft labels.
    """
    if not normalized:
        # Apply log softmax to logits to get the log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
    else:
        log_probs = torch.log(logits)
    # log_soft_targets = torch.log(soft_targets)

    # Calculate the KL divergence loss
    loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')     # the M-proj argmin_q KL(p||q), p: target; q: pred
    # loss = F.kl_div(log_probs, log_soft_targets, reduction='batchmean', log_target=True)

    return loss


if __name__ == '__main__':
    import dataloaders
    dataset = dataloaders.CGQA('/mnt/d/OneDrive - City University of Hong Kong - Student/datasets',
                               train=False, validation=True, download_flag=False, seed=0)
    dataset.load_dataset(9, train=False)
    aux = Auxiliary(dataset)


