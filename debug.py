
from typing import List, Dict, Any, Optional
import copy
import yaml
import json
import os
import pickle
from collections import OrderedDict
import argparse
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from learners.pmo_utils import Pool, draw_heatmap, draw_objs, cal_hv, cal_min_crowding_distance
from learners.slotmo import hungarian_algorithm


class Debugger:
    def __init__(self, level='DEBUG', args=None, exp_path=None, name=''):
        """args need to be dict"""
        assert (args is not None or exp_path is not None
                ), "args and exp_path should not be both None."

        self.levels = ['DEBUG', 'INFO']
        self.level = self.levels.index(level)   # 0 or 1
        self.args = copy.deepcopy(args)
        self.exp_path = exp_path
        if self.exp_path is not None:
            self.load_args(exp_path=self.exp_path)
        else:
            self.exp_path = self.args['log_dir']
        args = self.args
        self.name = name

        # for k, v in self.args.items():       # list to str
        #     if type(v) is list:
        #         # self.args[k] = ''.join(str(v).split(','))
        #         self.args[k] = str(v)

        self.dataset = self.args['dataset']         # CIFAR100, CGQA,...
        self.max_seed = self.args['repeat']
        self.max_task = self.args['max_task']
        self.save_path = os.path.join(self.exp_path, 'temp')

        self.output_args = []
        self.columns = []
        self._default_output_args()
        self.storage = {'samples': []}
        self.storage['results'] = {}
        self.storage['loss_df'] = {}

        # if use dataset
        self.trainer = None
        self.seed = 0
        self.learners = []
        self.tasks = None
        self.label_set = []      # [0,...,99]
        self.label_range = []
        self.single_label_datasets = {}
        self.single_label_dataloaders = {}

    def collect_results(self, max_task=-1, draw=False, use_dataset=False, label_range=range(0,10), select_ids='default'):
        # collect results
        self.storage['results'] = {}
        self.storage['loss_df'] = {}

        # todo: number of trainable parameters

        self.collect_AA_CA_FF(max_task)
        self.collect_CFST()
        self.collect_losses(draw=draw)

        if select_ids == 'default':
            select_ids = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]     # 3 samples

        if use_dataset:
            try:
                self.prepare_trainer(seed=0)
                self.load_samples(num_samples_per_class=2, label_range=label_range)
                self.collect_sample_results()       # obtain slots, prompts, attns,...
                self.collect_samples_attns_sim_per_img()
                for select_id in select_ids:
                    self.draw_attns(select_id=select_id, redraw=draw)
                    self.draw_slot_cos_sim(select_id=select_id, redraw=draw)
                self.collect_samples_slots_sim_per_img()
                self.collect_task_wise_attn_slot_change()
                self.collect_samples_weighted_slot_sim_per_class()
                self.draw_slot_weights(redraw=draw)
                self.draw_weighted_slot_similarity(redraw=draw)
                self.draw_weighted_mapped_slot_similarity(redraw=draw)
                self.draw_logit_similarity(redraw=draw)
                self.draw_prompt_selection(redraw=draw)
                self.draw_concept_similarity(redraw=draw)
            except Exception as e:
                # Print the error traceback
                traceback.print_exc()
                # if self.check_level('DEBUG'):
                print(f'Error collecting trainer results.')

    def loss_df(self):
        if 'loss_df' in self.storage:
            return self.storage['loss_df']
        else:
            return None

    def _default_output_args(self):
        # default params
        self.output_args = [
            'max_task', 'lr', 'slot_lr', 'prompt_param', 'larger_prompt_lr', 'batch_size']
        if self.args['lr_decreace_ratio'] != 1.0:
            self.output_args.append('lr_decrease_ratio')

        if self.args['learner_name'] == 'SLOTPrompt':
            # slot params
            self.output_args.extend(['n_slots', 'n_iters', 'slot_temp', 's2p_mode', 's2p_temp'])

    def default_columns(self, all_args=True):
        if all_args:
            return list(self.args.keys()), self.columns
        else:
            return self.output_args, self.columns

    def load_csv(self):
        csv_path = os.path.join(self.save_path, 'data.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.name = self.name
            return df
        else:
            return False

    def generate_df(self, column_info=None, save=False):
        """args and storage to value"""
        # form dict
        if column_info is None:
            column_info = self.default_columns()
        output_args, columns = column_info

        row = OrderedDict()
        row['name'] = self.name
        row['exp_path'] = self.exp_path
        row['dataset'] = self.dataset

        for output_arg in output_args:
            row[output_arg] = self.args.get(output_arg, '-')
        for res in columns:
            target = self.storage['results'].get(res, None)
            if target:
                row[res] = target['Mean']
                row[f'{res}(str)'] = F"{target['Mean']:.2f}$\pm${target['CI95']:.2f}({target['Std']:.2f})"

        df = pd.Series(data=row).to_frame().T
        if save:
            df.to_csv(os.path.join(self.save_path, 'data.csv'))

        return df

    def reset_seed(self, seed_value=0):
        # random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def prepare_trainer(self, seed=0):
        "seed to determine which run to load"
        from trainer import Trainer

        metric_keys = ['acc', 'time', ]
        save_keys = ['global', 'pt', 'pt-local']
        avg_metrics = {}
        for mkey in metric_keys:
            avg_metrics[mkey] = {}
            for skey in save_keys: avg_metrics[mkey][skey] = []

        self.seed = seed
        self.reset_seed(seed)
        trainer = Trainer(argparse.Namespace(**copy.deepcopy(self.args)), seed, metric_keys, save_keys)  # new trainer
        self.trainer = trainer
        self.learners = []
        for task_id in range(self.max_task):
            trainer.current_t_index = task_id
            task = trainer.tasks_logits[task_id]
            trainer.train_dataset.load_dataset(task_id, train=True)
            trainer.test_dataset.load_dataset(task_id, train=False)
            trainer.add_dim = len(task)
            if task_id > 0:
                try:
                    trainer.learner.model.prompt.process_task_count()
                except:
                    pass
            trainer.learner.last_valid_out_dim = trainer.learner.valid_out_dim
            trainer.learner.first_task = False
            trainer.learner.add_valid_output_dim(len(task))
            trainer.learner.pre_steps()
            trainer.learner.model.task_id = task_id
            trainer.learner.task_count = task_id
            trainer.learner.data_weighting(trainer.train_dataset)

            # load model
            model_save_dir = trainer.model_top_dir + '/models/repeat-' + str(trainer.seed + 1) + '/task-' + \
                             trainer.task_names[task_id] + '/'
            try:
                trainer.learner.load_model(model_save_dir)

                self.learners.append(copy.deepcopy(trainer.learner))
            except:
                pass

        # load tasks
        self.tasks = trainer.tasks_logits

        if self.check_level('DEBUG'):
            print(f'tasks: {self.tasks}.')
            # tasks
            dataset = trainer.test_dataset
            map_tuple_label_to_int = dataset.benchmark.label_info[1]
            class_mappings = dataset.benchmark.class_mappings[-1]
            map_related_int_to_tuple_label = {class_mappings[int_label]: str_label for str_label, int_label in
                                              map_tuple_label_to_int.items()}
            print(sorted(map_related_int_to_tuple_label.items()))
            # concepts
            label_concepts = dataset.get_concepts()
            map_int_concepts_label_to_str = dataset.benchmark.label_info[3]['map_int_concepts_label_to_str']
            print(f'label_concepts: {label_concepts}')
            print(sorted(map_int_concepts_label_to_str.items()))

        # load single-class-datasets
        dataset = trainer.test_dataset
        self.label_set = dataset.get_unique_labels()
        self.single_label_datasets = {
            label: copy.deepcopy(dataset.get_single_class_dataset(label)) for label in self.label_set}
        self.single_label_dataloaders = {
            label: DataLoader(self.single_label_datasets[label], batch_size=32, shuffle=False, drop_last=False,
                              num_workers=int(self.trainer.workers)) for label in self.label_set}
        if self.check_level('DEBUG'):
            print(f'label_set: {self.label_set}.')

    def load_samples(self, num_samples_per_class=2, label_range=None, reset_seed=True, draw=False):
        """load samples and store to storage['samples']"""
        if len(self.storage['samples']) > 0:        # already has samples
            return

        if reset_seed:
            self.reset_seed(self.seed)

        xs, ys, cs = [], [], []
        num_samples = num_samples_per_class

        if label_range is None:
            label_range = self.label_set
        self.label_range = list(label_range)
        for label in label_range:
            iterator = iter(self.single_label_dataloaders[label])
            sample = next(iterator)
            if len(sample) == 3:
                x, y, task = sample
                # send data to gpu
                x = x[:num_samples].cuda()
                y = y[:num_samples].cuda()
            else:
                x, y, c, task = sample
                # send data to gpu
                x = x[:num_samples].cuda()
                y = y[:num_samples].cuda()
                c = c[:num_samples].cuda()
                cs.append(c)

            xs.append(x)
            ys.append(y)

        x = torch.cat(xs)
        y = torch.cat(ys)
        if len(cs) > 0:
            c = torch.cat(cs)
        else:
            c = None

        self.storage['samples'] = [x, y, c]

        if self.check_level('DEBUG'):
            print(f'load_samples => {y}.')

        if draw:
            fig, axes = plt.subplots(len(self.label_set), num_samples, figsize=(num_samples*5,len(self.label_set)*5))
            for x_id in range(len(x)):
                xi = x_id // num_samples
                yi = x_id % num_samples
                label = y[x_id].item()
                ori_y_str = self.get_label_str(label)

                ori_x = unnormalize(x[x_id])
                axes[xi, yi].imshow(ori_x)
                axes[xi, yi].axis('off')
                axes[xi, yi].set_title(f'{x_id} {label} {ori_y_str}')

            self.savefig(fig, 'samples.png')

        return

    def collect_sample_results(self, reset_seed=True):
        """把attn,slot,prompts等等放到self.storage['attn']等里面, 要放到table里的放到'results'里用标准格式
        """
        if reset_seed:
            self.reset_seed(self.seed)

        self.storage['attns'] = []  # n_task*[bs, n, k]
        self.storage['slots'] = []  # n_task*[bs, k, h]
        self.storage['proms'] = []  # n_task*[bs, k, e, p, d]
        self.storage['weigs'] = []  # n_task*[bs, k]
        self.storage['wslos'] = []  # n_task*[bs, h]        # mapped slots @ weights
        self.storage['seles'] = []  # n_task*[bs, k, e, pp]
        self.storage['logis'] = []  # n_task*[bs, 100]
        self.storage['feats'] = []  # n_task*[bs, 768]

        x, y, c = self.storage['samples']
        num_tasks = len(self.learners)
        # trainer = prepare(trainer, task_id=-1, load=False)
        for task_id in range(num_tasks):
            learner = self.learners[task_id]
            model = learner.model
            prompt = model.prompt
            model.eval()

            # prompt.n_slots = 3
            # prompt.slot_attn[0].n_slots = 3

            with torch.no_grad():
                res = learner.forward(x, y)
                prompts = res['prompts']
                selections = res['selections']
                slot_weights = res['slot_weights']
                w_slots = res['w_slots']
                slots = res['slots']
                attn = res['attn']
                recon_loss = res['recon_loss']
                out = res['logits']     # [bs, 100]
                features = res['features']      # [bs, 768]

                bs, t, kk, e, p, d = prompts.shape      # [bs, t1, kk1, e5, p8, d768]
                bs, t, kk, e, pp = selections.shape  # [bs, t1, kk1, e5, pp50]   kk=1 use w-slot to select
                bs, t, h = w_slots.shape    # [bs, t1, h128]
                bs, t, k, h = slots.shape  # [bs, t1, k10, h128]
                bs, t, n, k = attn.shape  # [bs, t1, n196, k10]
                bs, t, k = slot_weights.shape   # [bs, t1, k10]
                assert t == 1

                prompts = prompts.reshape(bs, kk, e, p, d)
                attn = attn.reshape(bs, n, k)
                slots = slots.reshape(bs, k, h)
                w_slots = w_slots.reshape(bs, h)
                slot_weights = slot_weights.reshape(bs, k)
                selections = selections.reshape(bs, kk, e, pp)

                self.storage['attns'].append(attn)
                self.storage['slots'].append(slots)
                self.storage['proms'].append(prompts)
                self.storage['weigs'].append(slot_weights)
                self.storage['wslos'].append(w_slots)
                self.storage['seles'].append(selections)
                self.storage['logis'].append(out)
                self.storage['feats'].append(features)

        # align on pure slots
        anchor = self.storage['slots'][0].unsqueeze(2)  # [b, k, 1, h]
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        for task_id in range(1, len(self.storage['slots'])):
            ith_task_slots = self.storage['slots'][task_id].unsqueeze(1)  # [b, 1, k, h]
            sim = cos(anchor, ith_task_slots)  # [b, k, k]
            cost = 1 - sim
            _, index = hungarian_algorithm(cost)  # [b, 2, k]
            index = index[:, 1]  # [b, k]
            for sample_id in range(bs):
                self.storage['slots'][task_id][sample_id] = self.storage['slots'][task_id][sample_id][index[sample_id]]
                self.storage['attns'][task_id][sample_id] = self.storage['attns'][task_id][sample_id][:, index[sample_id]]
                # self.storage['proms'][task_id][sample_id] = self.storage['proms'][task_id][sample_id][index[sample_id]]
                self.storage['weigs'][task_id][sample_id] = self.storage['weigs'][task_id][sample_id][index[sample_id]]
                # collection_seles[task_id][sample_id] = collection_seles[task_id][sample_id][index[sample_id]]

    def collect_samples_attns_sim_per_img(self):
        # for every img, cal MAE(attn sim, I) to store in results
        img_attns = torch.cat(self.storage['attns'])  # [n_task*bs, n, k]
        cos_attn = nn.CosineSimilarity(dim=1, eps=1e-6)
        sim = cos_attn(img_attns.unsqueeze(2), img_attns.unsqueeze(3))  # [n_task*bs, k, k]
        eye = torch.eye(sim.shape[-1]).expand_as(sim).to(sim.device)
        attn_sim_mae = torch.abs(sim - eye)       # [n_task*bs, k, k]
        sim = sim.cpu().detach().numpy()
        attn_sim_mae = attn_sim_mae.cpu().detach().numpy()
        self.storage['results']['samples/per_img/attn_sim_mae'] = {
            'Details': sim, 'Mean': attn_sim_mae.mean(), 'Std': attn_sim_mae.std(),
            'CI95': 1.96 * (attn_sim_mae.std() / np.sqrt(np.prod(attn_sim_mae.shape)))}

        # extend columns
        self.columns.extend(['samples/per_img/attn_sim_mae'])

    def draw_attns(self, select_id, ax=None, redraw=False):
        ori_x = unnormalize(self.storage['samples'][0][select_id])

        label = self.storage['samples'][1][select_id].item()
        ori_y_str = self.get_label_str(label)

        name = f'{label}-{ori_y_str}-slot-attn.png'
        if not redraw and self.existfig(name):
            return

        # draw attn
        k = self.storage['attns'][0].shape[-1]      # 10
        n_row = len(self.storage['attns'])  # n_tasks
        n_column = k

        save = False
        fig = None
        if ax is None:
            fig, ax = plt.subplots(n_row, n_column, figsize=(n_column * 5, n_row * 5))
            save = True
        # otherwise, ax is a list of axis to fill in the imgs

        for task_id in range(n_row):
            attns = self.storage['attns'][task_id][select_id]  # [n196, k10]
            # slots = collection_slots[task_id][select_id]    # [k10, h128]

            for slot_id in range(k):
                xi = (slot_id + k * task_id) // n_column
                yi = (slot_id + k * task_id) % n_column
                _attn = attns[:, slot_id]
                if n_row == 1:
                    visualize_att_map(_attn.cpu().numpy(), ori_x, grid_size=14, alpha=0.6, ax=ax[yi])
                    if xi == 0:
                        ax[yi].set_title(yi, fontsize=50)
                    if yi == 0:
                        ax[yi].set_ylabel(f't{xi}', fontsize=50)
                else:
                    visualize_att_map(_attn.cpu().numpy(), ori_x, grid_size=14, alpha=0.6, ax=ax[xi, yi])
                    if xi == 0:
                        ax[xi, yi].set_title(yi, fontsize=50)
                    if yi == 0:
                        ax[xi, yi].set_ylabel(f't{xi}', fontsize=50)

        if save:
            fig.suptitle(f'{self.name}-slot-attn: {label}, {ori_y_str}', fontsize=16)
            self.savefig(fig, name)

    def collect_samples_slots_sim_per_img(self):
        # for every img, cal MAE(slot sim, I) to store in results
        img_slots = torch.cat(self.storage['slots'])  # [n_task*bs, k, h]
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        sim = cos(img_slots.unsqueeze(1), img_slots.unsqueeze(2))  # [n_task*bs, k, k]
        eye = torch.eye(sim.shape[-1]).expand_as(sim).to(sim.device)
        slot_sim_mae = torch.abs(sim - eye)  # [n_task*bs, k, k]
        sim = sim.cpu().detach().numpy()
        slot_sim_mae = slot_sim_mae.cpu().detach().numpy()
        self.storage['results']['samples/per_img/slot_sim_mae'] = {
            'Details': sim, 'Mean': slot_sim_mae.mean(), 'Std': slot_sim_mae.std(),
            'CI95': 1.96 * (slot_sim_mae.std() / np.sqrt(np.prod(slot_sim_mae.shape)))}

        # extend columns
        self.columns.extend(['samples/per_img/slot_sim_mae'])

    def draw_slot_cos_sim(self, select_id=0, ax=None, redraw=False):
        # global map of slot cossim
        k = self.storage['slots'][0].shape[1]      # 10

        label = self.storage['samples'][1][select_id].item()
        ori_y_str = self.get_label_str(label)
        name = f'{label}-{ori_y_str}-slot-cos-sim.png'
        if not redraw and self.existfig(name):
            return

        save = False
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(k * 1.5, k * 1.5))
            save = True

        _slotss = []
        for task_id in range(len(self.storage['slots'])):
            slots = self.storage['slots'][task_id][select_id]  # [k30, d128]
            _slotss.append(slots)

        _slotss = torch.cat(_slotss)  # [k30*3, d128]
        sim = F.cosine_similarity(_slotss.unsqueeze(0), _slotss.unsqueeze(1), dim=-1)
        # sim = -torch.norm(p.unsqueeze(0) - p.unsqueeze(1), dim=2)       # l2 -> [20, 20]
        # sim = torch.softmax(sim, dim=1)
        sim = sim.cpu().numpy()

        draw_heatmap(sim, verbose=False, ax=ax, fmt=".3f")
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        # ax.set_xticks([])
        # ax.set_yticks([])

        # save
        if save:
            fig.suptitle(f'{self.name}-slot-cos-sim: {label}, {ori_y_str}', fontsize=16)
            self.savefig(fig, name)

    def collect_task_wise_attn_slot_change(self):
        """After Hungarian match, attn differences and slot differences"""
        fst_task_attns = self.storage['attns'][0]    # [bs, n, k]
        fst_task_slots = self.storage['slots'][0]    # [bs, k, h]
        attn_maes = []
        slot_maes = []
        for task_id in range(1, len(self.storage['attns'])):
            task_attns = self.storage['attns'][task_id]
            attn_maes.append(torch.nn.functional.l1_loss(fst_task_attns, task_attns, reduction='none'))
            task_slots = self.storage['slots'][task_id]
            slot_maes.append(torch.nn.functional.l1_loss(fst_task_slots, task_slots, reduction='none'))

        if len(attn_maes) > 0:
            attn_maes = torch.stack(attn_maes)      # [n_task-1, bs, n, k]
            attn_maes = attn_maes.cpu().detach().numpy()
            self.storage['results']['samples/task_wise/attn_mae'] = {
                'Details': attn_maes, 'Mean': attn_maes.mean(), 'Std': attn_maes.std(),
                'CI95': 1.96 * (attn_maes.std() / np.sqrt(np.prod(attn_maes.shape)))}

            slot_maes = torch.stack(slot_maes)      # [n_task-1, bs, k, h]
            slot_maes = slot_maes.cpu().detach().numpy()
            self.storage['results']['samples/task_wise/slot_mae'] = {
                'Details': slot_maes, 'Mean': slot_maes.mean(), 'Std': slot_maes.std(),
                'CI95': 1.96 * (slot_maes.std() / np.sqrt(np.prod(slot_maes.shape)))}

            # extend columns
            self.columns.extend(['samples/task_wise/attn_mae', 'samples/task_wise/slot_mae'])

    def collect_samples_weighted_slot_sim_per_class(self, mapped=True):
        """refer to intra-class-consistency-reg"""
        slots = self.storage['slots']       # n_tasks * [bs, k, h]
        slot_weights = self.storage['weigs']    # n_tasks * [bs, k]
        w_slots = self.storage['wslos']  # n_tasks * [bs, d128]
        ys = self.storage['samples'][1]     # [bs]
        unique_ys = torch.unique(ys)
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        task_sims = []
        for task_id in range(len(self.storage['slots'])):
            cls_sims = []
            if mapped:
                weighted_slots = w_slots[task_id]
            else:
                weighted_slots = torch.einsum('bkh,bk->bh', slots[task_id], slot_weights[task_id])
            for label in unique_ys:
                selected_idxs = torch.where(ys == label)[0]
                selected_w_slots = weighted_slots[selected_idxs]    # [2, h]
                sim = cos(selected_w_slots.unsqueeze(0), selected_w_slots.unsqueeze(1))  # [2, 2]
                cls_sims.append(sim)
            cls_sims = torch.stack(cls_sims)    # [n_cls, 2, 2]
            task_sims.append(cls_sims)
        task_sims = torch.stack(task_sims)  # [n_tasks, n_cls, 2, 2]
        task_sims = task_sims.cpu().detach().numpy()
        self.storage['results']['samples/per_cls/w_slot_sim'] = {
            'Details': task_sims, 'Mean': task_sims.mean(), 'Std': task_sims.std(),
            'CI95': 1.96 * (task_sims.std() / np.sqrt(np.prod(task_sims.shape)))}

        # extend columns
        self.columns.extend(['samples/per_cls/w_slot_sim'])

    def draw_slot_weights(self, ax=None, redraw=False):
        # draw ws
        name = f'slot-selection.png'
        if not redraw and self.existfig(name):
            return

        n_row = 1
        n_column = len(self.storage['weigs'])  # n_tasks

        save = False
        fig = None
        if ax is None:
            fig, ax = plt.subplots(n_row, n_column, figsize=(n_column * 5, n_row * 5))
            save = True

        for task_id in range(n_column):
            ws = self.storage['weigs'][task_id]  # [bs, k10]
            yi = task_id
            if n_column == 1:
                draw_heatmap(ws.cpu().numpy(), verbose=True, ax=ax, fmt=".2f")
                ax.set_title(f't{task_id}', fontsize=16)
            else:
                draw_heatmap(ws.cpu().numpy(), verbose=True, ax=ax[yi], fmt=".2f")
                ax[yi].set_title(f't{task_id}', fontsize=16)

        # save
        if save:
            fig.suptitle(f'{self.name}-slot-selection', fontsize=16)
            self.savefig(fig, name)

    def draw_weighted_slot_similarity(self, ax=None, redraw=False):
        """obtain w_slots -> according to slot-logit-sim-reg-mode obtain sim"""
        name = f'weighted-slot-sim.png'
        if not redraw and self.existfig(name):
            return

        n_row = 2
        n_column = len(self.storage['weigs'])  # n_tasks

        save = False
        fig = None
        if ax is None:
            fig, ax = plt.subplots(n_row, n_column, figsize=(n_column * 5, n_row * 5))
            save = True

        for task_id in range(n_column):
            ws = self.storage['weigs'][task_id]  # [bs, k10]
            slots = self.storage['slots'][task_id]      # [bs, k10, d128]
            weighted_slot = torch.einsum('bkd,bk->bd', slots, ws)       # [bs, d128]

            if 'cos' in self.args['slot_logit_similar_reg_mode']:
                cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                slot_sim = cos(weighted_slot.unsqueeze(1), weighted_slot.unsqueeze(0)
                               ) * self.args['slot_logit_similar_reg_slot_temp']  # [bs, bs]
                normed_slot_sim = (slot_sim - slot_sim.min(dim=-1, keepdim=True)[0]) / (
                        slot_sim.max(dim=-1, keepdim=True)[0] - slot_sim.min(dim=-1, keepdim=True)[0] + 1e-10)
                # minmax over row to make them positive
                normed_slot_sim = normed_slot_sim / normed_slot_sim.sum(dim=-1, keepdim=True)  # l1-norm
            else:
                slot_sim = torch.matmul(weighted_slot, weighted_slot.t()) * (
                        self.args['slot_logit_similar_reg_slot_temp'] * weighted_slot.shape[-1] ** -0.5)
                normed_slot_sim = F.softmax(slot_sim, dim=-1)

            yi = task_id
            if n_column == 1:
                draw_heatmap(slot_sim.cpu().numpy(), verbose=False, ax=ax[0], fmt=".2f")
                draw_heatmap(normed_slot_sim.cpu().numpy(), verbose=False, ax=ax[1], fmt=".2f")
                ax[0].set_title(f't{task_id}', fontsize=16)
                if yi == 0:
                    ax[0].set_ylabel('sim')
                    ax[1].set_ylabel('norm')
            else:
                draw_heatmap(slot_sim.cpu().numpy(), verbose=False, ax=ax[0, yi], fmt=".2f")
                draw_heatmap(normed_slot_sim.cpu().numpy(), verbose=False, ax=ax[1, yi], fmt=".2f")
                ax[0, yi].set_title(f't{task_id}', fontsize=16)
                if yi == 0:
                    ax[0, yi].set_ylabel('sim')
                    ax[1, yi].set_ylabel('norm')

        if save:
            fig.suptitle(f'{self.name}-weighted-slot-sim', fontsize=16)
            self.savefig(fig, name)

    def draw_weighted_mapped_slot_similarity(self, ax=None, redraw=False):
        """obtain w_slots -> according to slot-logit-sim-reg-mode obtain sim"""
        name = f'weighted-mapped-slot-sim.png'
        if not redraw and self.existfig(name):
            return

        n_row = 2
        n_column = len(self.storage['wslos'])  # n_tasks

        save = False
        fig = None
        if ax is None:
            fig, ax = plt.subplots(n_row, n_column, figsize=(n_column * 5, n_row * 5))
            save = True

        for task_id in range(n_column):
            weighted_slot = self.storage['wslos'][task_id]  # [bs, d128]

            if 'cos' in self.args['slot_logit_similar_reg_mode']:
                cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                slot_sim = cos(weighted_slot.unsqueeze(1), weighted_slot.unsqueeze(0)
                               ) * self.args['slot_logit_similar_reg_slot_temp']  # [bs, bs]
                # normed_slot_sim = slot_sim / slot_sim.sum(dim=-1, keepdim=True)  # l1-norm
                normed_slot_sim = (slot_sim - slot_sim.min(dim=-1, keepdim=True)[0]) / (
                        slot_sim.max(dim=-1, keepdim=True)[0] - slot_sim.min(dim=-1, keepdim=True)[0] + 1e-10)
                # minmax over row to make them positive
                normed_slot_sim = normed_slot_sim / normed_slot_sim.sum(dim=-1, keepdim=True)  # l1-norm
            else:
                slot_sim = torch.matmul(weighted_slot, weighted_slot.t()) * (
                        self.args['slot_logit_similar_reg_slot_temp'] * weighted_slot.shape[-1] ** -0.5)
                normed_slot_sim = F.softmax(slot_sim, dim=-1)

            yi = task_id
            if n_column == 1:
                draw_heatmap(slot_sim.cpu().numpy(), verbose=False, ax=ax[0], fmt=".2f")
                draw_heatmap(normed_slot_sim.cpu().numpy(), verbose=False, ax=ax[1], fmt=".2f")
                ax[0].set_title(f't{task_id}', fontsize=16)
                if yi == 0:
                    ax[0].set_ylabel('sim')
                    ax[1].set_ylabel('norm')
            else:
                draw_heatmap(slot_sim.cpu().numpy(), verbose=False, ax=ax[0, yi], fmt=".2f")
                draw_heatmap(normed_slot_sim.cpu().numpy(), verbose=False, ax=ax[1, yi], fmt=".2f")
                ax[0, yi].set_title(f't{task_id}', fontsize=16)
                if yi == 0:
                    ax[0, yi].set_ylabel('sim')
                    ax[1, yi].set_ylabel('norm')

        if save:
            fig.suptitle(f'{self.name}-weighted-mapped-slot-sim', fontsize=16)
            self.savefig(fig, name)

    def draw_logit_similarity(self, ax=None, redraw=False):
        """obtain logits -> according to slot-logit-sim-reg-mode obtain sim"""
        name = f'logit-sim.png'
        if not redraw and self.existfig(name):
            return

        n_row = 2
        n_column = len(self.storage['weigs'])  # n_tasks

        save = False
        fig = None
        if ax is None:
            fig, ax = plt.subplots(n_row, n_column, figsize=(n_column * 5, n_row * 5))
            save = True

        for task_id in range(n_column):
            logits = self.storage['logis'][task_id]     # [bs, 100]
            task_mask = self.label_range
            masked_logits = logits[:, task_mask]        # [bs, 10]
            if self.check_level('DEBUG'):
                print(f'draw_logit_similarity => task_mask: {task_mask}.')
                print(f'draw_logit_similarity => masked_logits[0]: {masked_logits[0]}.')

            logit_sim = torch.matmul(logits, logits.t()) * (
                    self.args['slot_logit_similar_reg_temp'] * (logits.shape[-1] ** -0.5))
            logit_sim_softmax = F.softmax(logit_sim, dim=-1)

            yi = task_id
            if n_column == 1:
                draw_heatmap(logit_sim.cpu().numpy(), verbose=False, ax=ax[0], fmt=".2f")
                draw_heatmap(logit_sim_softmax.cpu().numpy(), verbose=False, ax=ax[1], fmt=".2f")
                ax[0].set_title(f't{task_id}', fontsize=16)
                if yi == 0:
                    ax[0].set_ylabel('sim')
                    ax[1].set_ylabel('softmax')
            else:
                draw_heatmap(logit_sim.cpu().numpy(), verbose=False, ax=ax[0, yi], fmt=".2f")
                draw_heatmap(logit_sim_softmax.cpu().numpy(), verbose=False, ax=ax[1, yi], fmt=".2f")
                ax[0, yi].set_title(f't{task_id}', fontsize=16)
                if yi == 0:
                    ax[0, yi].set_ylabel('sim')
                    ax[1, yi].set_ylabel('softmax')

        if save:
            fig.suptitle(f'{self.name}-logit-sim', fontsize=16)
            self.savefig(fig, name)

    def draw_concept_similarity(self, ax=None, redraw=False):
        name = f'concept-sim.png'
        if not redraw and self.existfig(name):
            return

        if self.storage['samples'][2] is None:      # dataset no concept
            return

        n_row = 2
        n_column = len(self.storage['weigs'])  # n_tasks

        save = False
        fig = None
        if ax is None:
            fig, ax = plt.subplots(n_row, n_column, figsize=(n_column * 5, n_row * 5))
            save = True

        for task_id in range(n_column):
            concepts = self.storage['samples'][2][:, 0]  # [bs, 21]
            if self.check_level('DEBUG'):
                print(f'draw_concept_similarity => concepts: {concepts.shape}.')

            cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
            if 'cos' in self.args['concept_similar_reg_mode']:
                concept_sim = cos(concepts.unsqueeze(1), concepts.unsqueeze(0)) * 1  # [bs, bs]
            else:
                concept_sim = cos(concepts.unsqueeze(1), concepts.unsqueeze(0))
                concept_sim = concept_sim / concept_sim.sum(dim=-1, keepdim=True)    # l1-norm
            concept_sim_softmax = F.softmax(concept_sim, dim=-1)

            yi = task_id
            if n_column == 1:
                draw_heatmap(concept_sim.cpu().numpy(), verbose=False, ax=ax[0], fmt=".2f")
                draw_heatmap(concept_sim_softmax.cpu().numpy(), verbose=False, ax=ax[1], fmt=".2f")
                ax[0].set_title(f't{task_id}', fontsize=16)
                if yi == 0:
                    ax[0].set_ylabel('sim')
                    ax[1].set_ylabel('softmax')
            else:
                draw_heatmap(concept_sim.cpu().numpy(), verbose=False, ax=ax[0, yi], fmt=".2f")
                draw_heatmap(concept_sim_softmax.cpu().numpy(), verbose=False, ax=ax[1, yi], fmt=".2f")
                ax[0, yi].set_title(f't{task_id}', fontsize=16)
                if yi == 0:
                    ax[0, yi].set_ylabel('sim')
                    ax[1, yi].set_ylabel('softmax')

        if save:
            fig.suptitle(f'{self.name}-concept-sim', fontsize=16)
            self.savefig(fig, name)

    def draw_prompt_selection(self, select_id=None, ax=None, redraw=False):
        """n_column=n_layer, n_row=1. each ax contains """
        if select_id is not None:
            label = self.storage['samples'][1][select_id].item()
            ori_y_str = self.get_label_str(label)
            name = f'{label}-{ori_y_str}-prompt-selection.png'
        else:
            name = f'prompt-selection.png'
        if not redraw and self.existfig(name):
            return

        bs, kk, e, pp = self.storage['seles'][-1].shape       # n_task*[bs, kk1, e5, pp100?]
        # pp is max_selection_dim
        n_row = len(self.storage['seles'])
        n_column = e
        save = False
        fig = None
        if ax is None:
            fig, ax = plt.subplots(n_row, n_column, figsize=(n_column * 5, n_row * 5))
            save = True

        for task_id in range(len(self.storage['seles'])):
            bs, kk, e, pp = self.storage['seles'][task_id].shape       # update pp since different task different.
            if select_id is not None:
                selections = self.storage['seles'][task_id][select_id]  # [kk1, e5, pp100?]
            else:
                selections = self.storage['seles'][task_id].reshape(bs*kk, e, pp)

            for layer_id in range(e):
                xi = task_id
                yi = layer_id
                _selection = selections[:, layer_id]    # [kk, pp]

                if len(self.storage['seles']) == 1:
                    draw_heatmap(_selection.cpu().numpy(), verbose=False, ax=ax[yi], fmt=".3f")
                    if xi == 0:
                        ax[yi].set_title(f'l{yi}', fontsize=16)
                    if yi == 0:
                        ax[yi].set_ylabel(f't{xi}', fontsize=16)
                else:
                    draw_heatmap(_selection.cpu().numpy(), verbose=False, ax=ax[xi, yi], fmt=".3f")
                    if xi == 0:
                        ax[xi, yi].set_title(f'l{yi}', fontsize=16)
                    if yi == 0:
                        ax[xi, yi].set_ylabel(f't{xi}', fontsize=16)

        # save
        if save:
            if select_id is not None:
                label = self.storage['samples'][1][select_id].item()
                ori_y_str = self.get_label_str(label)
                fig.suptitle(f'{self.name}-prompt-selection: {label}, {ori_y_str}', fontsize=16)
                self.savefig(fig, name)
            else:
                fig.suptitle(f'{self.name}-prompt-selection', fontsize=16)
                self.savefig(fig, name)

    def get_label_str(self, related_label):
        try:
            ori_y = self.trainer.test_dataset.benchmark.original_classes_in_exp.flatten()[related_label]
            ori_y_str = self.trainer.test_dataset.benchmark.label_info[2][ori_y]
        except:
            ori_y_str = ''

        return ori_y_str

    def collect_AA_CA_FF(self, max_task=-1, weighting=False):
        # pt
        file = os.path.join(self.exp_path, 'results-acc', 'pt.yaml')
        try:
            data_yaml = yaml.load(open(file, 'r'), Loader=yaml.Loader)
            data = np.array(data_yaml['history'])  # [n_tsk, n_tsk, n_run]
        except:
            if self.check_level('DEBUG'):
                print(f'File not find: {file}.')
            return
            # data = np.zeros((2,2,2))
        # example: data[:,:,0]
        # [[94.2 84.4 78.7 69.3 69.  66.5 62.  60.6 49.4 44.1]
        #  [ 0.  80.7 74.6 70.1 65.2 63.2 57.1 57.2 51.9 44.8]
        #  [ 0.   0.  79.1 74.2 74.2 66.7 61.9 61.3 55.9 48.3]
        #  [ 0.   0.   0.  75.8 72.4 62.4 59.7 57.5 55.3 50.9]
        #  [ 0.   0.   0.   0.  58.6 54.1 53.  48.6 45.  40.1]
        #  [ 0.   0.   0.   0.   0.  70.  66.3 64.9 61.  56. ]
        #  [ 0.   0.   0.   0.   0.   0.  74.5 67.9 66.5 64.4]
        #  [ 0.   0.   0.   0.   0.   0.   0.  62.3 55.5 51.7]
        #  [ 0.   0.   0.   0.   0.   0.   0.   0.  78.2 75.4]
        #  [ 0.   0.   0.   0.   0.   0.   0.   0.   0.  55. ]]

        if max_task == -1:
            max_task = data.shape[1]
        self.args['max_task'] = max_task        # visually change the number of finished tasks

        if max_task > data.shape[1]:    # haven't finished yet
            return

        if weighting and max_task > 1:
            '''weighting according to first_split_size and other_split_size'''
            weighting = np.array(
                [self.args['first_split_size'], *[self.args['other_split_size'] for _ in range(max_task - 1)]])
        else:
            weighting = np.array([1 for _ in range(max_task)])

        data_aa = data[:max_task, max_task - 1]    # [n_tsk, n_run]
        AA = np.average(data_aa, weights=weighting, axis=0)     # [n_run]
        data_cu = np.array([data[i, i:max_task].mean(axis=0) for i in range(max_task)])  # [n_tsk, n_run]
        CA = np.average(data_cu, weights=weighting, axis=0)     # [n_run]
        data_ff = np.array([data[i, i] - data[i, max_task-1] for i in range(max_task - 1)])  # [n_tsk-1, n_run]
        try:        # if only 1 task, weighting null error
            FF = np.average(data_ff, weights=weighting[:-1], axis=0)     # [n_run]
        except Exception as e:
            FF = np.array([np.nan])
        self.storage['results']['AA'] = {
            'Details': AA, 'Mean': AA.mean(), 'Std': AA.std(), 'CI95': 1.96 * (AA.std() / np.sqrt(len(AA)))}
        self.storage['results']['CA'] = {
            'Details': CA, 'Mean': CA.mean(), 'Std': CA.std(), 'CI95': 1.96 * (CA.std() / np.sqrt(len(CA)))}
        self.storage['results']['FF'] = {
            'Details': FF, 'Mean': FF.mean(), 'Std': FF.std(), 'CI95': 1.96 * (FF.std() / np.sqrt(len(FF)))}

        # local pt
        try:
            file = os.path.join(self.exp_path, 'results-acc', 'pt-local.yaml')
            data_yaml = yaml.load(open(file, 'r'), Loader=yaml.Loader)
            data = np.array(data_yaml['history'])  # [n_tsk, n_tsk, n_run]
        except:
            if self.check_level('DEBUG'):
                print(f'File not find: {file}.')
            return
            # data = np.zeros((2,2,2))

        data_aa = data[:max_task, max_task - 1]    # [n_tsk, n_run]
        AA = np.average(data_aa, weights=weighting, axis=0)     # [n_run]
        data_cu = np.array([data[i, i:max_task].mean(axis=0) for i in range(max_task)])  # [n_tsk, n_run]
        CA = np.average(data_cu, weights=weighting, axis=0)     # [n_run]
        data_ff = np.array([data[i, i] - data[i, max_task-1] for i in range(max_task - 1)])  # [n_tsk-1, n_run]
        try:
            FF = np.average(data_ff, weights=weighting[:-1], axis=0)     # [n_run]
        except Exception as e:
            FF = np.array([np.nan])
        self.storage['results']['l-AA'] = {
            'Details': AA, 'Mean': AA.mean(), 'Std': AA.std(), 'CI95': 1.96 * (AA.std() / np.sqrt(len(AA)))}
        self.storage['results']['l-CA'] = {
            'Details': CA, 'Mean': CA.mean(), 'Std': CA.std(), 'CI95': 1.96 * (CA.std() / np.sqrt(len(CA)))}
        self.storage['results']['l-FF'] = {
            'Details': FF, 'Mean': FF.mean(), 'Std': FF.std(), 'CI95': 1.96 * (FF.std() / np.sqrt(len(FF)))}

        # extend columns
        self.columns.extend(['AA', 'l-AA', 'CA', 'l-CA', 'FF', 'l-FF'])

    def collect_CFST(self):
        dataset = self.dataset
        mean_datas = {}
        for target in ['sys', 'pro', 'sub', 'non', 'noc'] if dataset == 'CGQA' else ['sys', 'pro', 'non', 'noc']:
            file = os.path.join(self.exp_path, 'results-acc', f'global-{target}.yaml')
            try:
                data_yaml = yaml.load(open(file, 'r'), Loader=yaml.Loader)
                data = np.array(data_yaml['mean'])  # [50]
            except:
                if self.check_level('DEBUG'):
                    print(f'File not find: {file}.')
                return
                # data = np.zeros((2))

            mean_data = data.mean()
            mean_datas[target] = mean_data
            std_data = data.std()
            ci95_data = 1.96 * (std_data / np.sqrt(len(data)))

            # print(f'{target}: {mean_data:.2f}$\pm${ci95_data:.2f}, std:{std_data:.2f}')
            self.storage['results'][target] = {'Details': data, 'Mean': mean_data, 'Std': std_data, 'CI95': ci95_data}

        if dataset == 'CGQA':
            hn = 3 / (1 / mean_datas['sys'] + 1 / mean_datas['pro'] + 1 / mean_datas['sub'])
            hr = 2 / (1 / mean_datas['non'] + 1 / mean_datas['noc'])
            ha = 5 / (1 / mean_datas['sys'] + 1 / mean_datas['pro'] + 1 / mean_datas['sub'] + 1 / mean_datas[
                'non'] + 1 / mean_datas['noc'])
        else:
            hn = 2 / (1 / mean_datas['sys'] + 1 / mean_datas['pro'])
            hr = 2 / (1 / mean_datas['non'] + 1 / mean_datas['noc'])
            ha = 4 / (1 / mean_datas['sys'] + 1 / mean_datas['pro'] + 1 / mean_datas['non'] + 1 / mean_datas['noc'])
        # print(f"Hn: {hn:.2f}; Hr: {hr:.2f}; Ha: {ha:.2f}")
        self.storage['results']['Hn'] = {'Details': [hn], 'Mean': hn, 'Std': 0, 'CI95': 0}
        self.storage['results']['Hr'] = {'Details': [hr], 'Mean': hr, 'Std': 0, 'CI95': 0}
        self.storage['results']['Ha'] = {'Details': [ha], 'Mean': ha, 'Std': 0, 'CI95': 0}

        # extend columns
        if self.dataset == 'CGQA':
            self.columns.extend(['sys', 'pro', 'sub', 'Hn', 'non', 'noc', 'Hr', 'Ha'])
        else:
            self.columns.extend(['sys', 'pro', 'Hn', 'non', 'noc', 'Hr', 'Ha'])  # no sub

    def load_log_data(self):
        # load log
        if self.check_level('DEBUG'):
            print(f'Load log data.')

        self.storage['log'] = {}
        max_task = self.max_task
        if max_task == -1:
            max_task = 10
        for seed in range(self.max_seed):
            try:
                for task in range(max_task):
                    file = os.path.join(self.exp_path, 'temp', f'train_log_seed{seed}_t{task}.pkl')
                    data = pickle.load(open(file, 'rb'))     # {'scaler': df, 'mo': df}
                    # df = data['scaler']
                    # df_mo = data['mo']
                    if seed not in self.storage['log'].keys():
                        self.storage['log'][seed] = {}
                    self.storage['log'][seed][task] = data
            except:
                if self.check_level('INFO'):
                    print(f'File not find: {file}.')
        if self.check_level('INFO'):
            current_seed = len(self.storage["log"]) - 1
            current_task = len(self.storage["log"][current_seed]) - 1
            print(f'Load seed {current_seed}, task {current_task}.')

    def collect_losses(self, draw=False):
        if 'loss_df' not in self.storage.keys():
            self.storage['loss_df'] = dict()
        res = self.storage['loss_df']
        if 'log' not in self.storage:
            self.load_log_data()

        max_seed = self.args['repeat']
        max_task = self.args['max_task']
        finished_seed = len(self.storage['log'])
        finished_task = len(self.storage['log'][finished_seed-1])
        if max_task == -1:
            max_task = finished_task

        candidate_keys = list(set(self.storage['log'][finished_seed-1][finished_task-1]['scaler'].Tag))

        # keys and put coeff to output_args
        keys = []
        # training and validation losses
        if 'val_recon_loss' in candidate_keys:
            keys.extend(['val_recon_loss'])
        if 'val_acc' in candidate_keys:
            keys.extend(['val_acc'])
        if 'loss/slot_recon_loss' in candidate_keys:
            keys.extend(['loss/slot_recon_loss'])
        if 'loss/ce_loss' in candidate_keys:
            keys.extend(['loss/ce_loss'])

        # slot reg losses
        if self.args['learner_name'] == 'SLOTPrompt':
            if self.args.get('use_intra_consistency_reg', False):
                self.output_args.extend(['intra_consistency_reg_mode', 'intra_consistency_reg_coeff'])
                keys.extend(['loss/intra_consistency_loss'])
            if self.args.get('use_slot_ortho_reg', False):
                self.output_args.extend(['slot_ortho_reg_mode', 'slot_ortho_reg_coeff'])
                keys.extend(['loss/slot_ortho_loss'])

        # prompt reg losses
        if self.args.get('use_weight_reg', False):
            self.output_args.extend(['weight_reg_mode', 'weight_reg_coeff'])
            keys.extend(['loss/s2p_loss'])
        if self.args.get('use_selection_onehot_reg', False):
            self.output_args.extend(['selection_onehot_reg_mode', 'selection_onehot_reg_coeff'])
            keys.extend(['loss/selection_onehot_loss'])
        if self.args.get('use_selection_slot_similar_reg', False):
            self.output_args.extend(['selection_slot_similar_reg_mode', 'selection_slot_similar_reg_coeff'])
            keys.extend(['loss/selection_ortho_loss'])
        if self.args.get('use_prompt_concept_alignment_reg', False):
            self.output_args.extend(['prompt_concept_alignment_reg_coeff'])
            keys.extend(['loss/prompt_concept_alignment_loss'])
        if self.args.get('concept_weight', False):
            self.output_args.extend(['concept_similar_reg_mode', 'concept_similar_reg_coeff', 'concept_similar_reg_temp'])
            keys.extend(['loss/concept_similar_reg'])
        if self.args.get('use_old_samples_for_reg', False):
            self.output_args.extend(['use_old_samples_for_reg'])
        if self.args.get('use_slot_logit_similar_reg', False):
            self.output_args.extend(['slot_logit_similar_reg_mode', 'slot_logit_similar_reg_coeff',
                                     'slot_logit_similar_reg_temp', 'slot_logit_similar_reg_slot_temp'])
            keys.extend(['loss/slot_logit_similar_reg'])

        # extend columns
        self.columns.extend(keys)

        if self.check_level('INFO'):
            print(f'Collect regs: {keys}.')

        # cat df
        nrows, ncols = 1, len(keys)
        if draw and nrows > 0 and ncols > 0:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*5))   # sharex=True, sharey=True
            fig.suptitle(f'{self.name}', fontsize=16)
        else:
            fig, axes = None, None
        for key_idx, key in enumerate(keys):
            dfs = []
            for seed in range(finished_seed):
                task_offset = 0
                for task in range(finished_task):
                    df = self.storage['log'][seed][task]['scaler']
                    t_df = copy.deepcopy(df[df.Tag == key])
                    # shift Idx with task
                    unique_idx = list(set(t_df.Idx))
                    num_idx = np.max(unique_idx) - np.min(unique_idx) + 1
                    t_df.Idx = t_df.Idx + task_offset
                    task_offset += num_idx
                    dfs.append(t_df)
            dfs = pd.concat(dfs, ignore_index=True)

            max_idx = np.max(list(set(dfs.Idx)))
            t_series = dfs[dfs.Idx == max_idx].Value
            # collect last Idx's value to show in the table
            self.storage['results'][key] = {'Details': t_series, 'Mean': t_series.mean(),
                                            'Std': t_series.std(),
                                            'CI95': 1.96 * (t_series.std() / np.sqrt(len(t_series)))}

            res[key] = dfs

            if draw and nrows > 0 and ncols > 0:
                if nrows == 1 and ncols == 1:
                    ax = axes
                else:
                    ax = axes[key_idx]
                self.draw(dfs, ax=ax, title=f'{key}')

        if draw and nrows > 0 and ncols > 0:
            self.savefig(fig, 'loss.png')

        return res

    @staticmethod
    def draw(df, ax=None, title=None):
        """df contains Idx, Value"""
        if ax is None:
            fig, ax = plt.subplots()

        ax.grid(True)
        sns.lineplot(df, x='Idx', y='Value', ax=ax)
        if title:
            ax.set_title(title)

    def savefig(self, fig, file_name):
        file_name = '-'.join(file_name.split('/'))
        fig.savefig(os.path.join(self.save_path, file_name), bbox_inches='tight', dpi=100)

    def existfig(self, file_name):
        file_name = '-'.join(file_name.split('/'))
        return os.path.exists(os.path.join(self.save_path, file_name))

    def draw_scaler(self, key, seed, task, ax=None, title=False):
        if 'log' not in self.storage:
            self.load_log_data()
        if ax is None:
            fig, ax = plt.subplots()
        ax.grid(True)

        df = self.storage['log'][seed][task]['scaler']
        t_df = df[df.Tag == key]

        sns.lineplot(t_df, x='Idx', y='Value', ax=ax)
        if title:
            ax.set_title(f'{key}-run{seed}-t{task}')

        return ax

    def load_args(self, exp_path):
        print(f'Load args from {exp_path}.')
        self.exp_path = exp_path
        path = os.path.join(exp_path, 'args.yaml')
        args = yaml.load(open(path, 'r'), Loader=yaml.Loader)
        args['gpuid'] = [0]
        args['debug_mode'] = 1
        self.args = args

    def check_level(self, level):
        thres = self.levels.index(level)
        return thres >= self.level

    """
    RUNTIME FUNCTION BELOW
    """

    def print_prototype_change(self, model: nn.Module, i, writer: Optional[SummaryWriter] = None):
        """

        Args:
            model: target model contain prototypes
            writer: None or obj: writer
            i: iter index

        Returns:

        """
        level = self.levels.index('DEBUG')
        if level < self.level:
            return

        proto = model.selector.prototypes.detach().cpu().numpy()
        if 'proto' in self.storage:
            old_proto = self.storage['proto']
        else:
            old_proto = 0
        dif = np.linalg.norm((proto - old_proto).ravel(), 2)  # l2 distance
        self.storage['proto'] = proto

        print(f'proto diff (l2) is {dif}.\nmov_avg_alpha is {model.selector.mov_avg_alpha.item()}.')
        if writer is not None:
            writer.add_scalar('params/cluster-centers-dif', dif, i + 1)
            writer.add_scalar('params/mov_avg_alpha', model.selector.mov_avg_alpha.item(), i + 1)

    def print_grad(self, model: nn.Module, key=None, prefix=''):
        """

        Args:
            model: target model
            key: for parameter name and also for storage name for diff
            prefix: print prefix

        Returns:

        """
        level = self.levels.index('DEBUG')
        if level < self.level:
            return

        vs = []
        with torch.no_grad():
            for k, v in model.named_parameters():
                if (key is None) or (key in k):
                    if v.grad is not None:
                        vs.append(v.grad.flatten())
            vs = torch.cat(vs).detach().cpu().numpy()
            if key in self.storage:
                dif = vs - self.storage[key]
            else:
                dif = vs
            self.storage[key] = vs
            dif = np.abs(dif)
            print(f'{prefix}mean abs grad diff for {key} is {np.mean(dif)}.')

    def save_log(self, log, log_name):
        """

        Args:
            log: dict
            log_name: path to save log
        """
        print('=> Saving log to:', log_name)
        with open(log_name, 'wb') as f:
            pickle.dump(log, f)
        print('=> Save Done')

    def write_pool(self, pool: Pool, i, writer: Optional[SummaryWriter] = None, prefix='pool'):
        """

        Args:
            pool: pool after call buffer2cluster()
            i: iter index
            writer: None or obj: writer
            prefix:

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        print(f'iter {i}: {prefix} num_cls info: '
              f'{[f"{idx}[{len(sim)}]" for idx, sim in enumerate(pool.current_similarities())]}.')

        if writer is not None:
            '''all images and img_sims and class_sims '''
            images = pool.current_images(single_image=True)
            for cluster_id, cluster in enumerate(images):
                if len(cluster) > 0:
                    writer.add_image(f"{prefix}-img/{cluster_id}", cluster, i + 1)
            similarities = pool.current_similarities(image_wise=True)
            class_similarities = pool.current_similarities()
            for cluster_id, (img_sim, cls_sim) in enumerate(zip(similarities, class_similarities)):
                if len(img_sim) > 0:
                    # img_sim [num_cls * [num_img, 8]]; cls_sim [num_cls * [8]]
                    sim = np.concatenate([
                        np.concatenate([img_sim[cls_idx],
                                        *[cls_sim[cls_idx][np.newaxis, :]] * max(10, len(img_sim[cls_idx]) // 2)])
                        for cls_idx in range(len(img_sim))
                    ])
                    figure = draw_heatmap(sim, verbose=False)
                    writer.add_figure(f"{prefix}-img-sim/{cluster_id}", figure, i + 1)

    def write_scaler(self, df, key, i, writer: Optional[SummaryWriter] = None, prefix='', inner=True):
        """

        Args:
            df: [Tag, Idx, Value]
            key: in Tag
            i:
            writer:
            prefix:
            inner: True, write image based on Idx

        Returns: last Idx's avg_value or -1

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        if key == 'all':
            keys = list(set(df.Tag))
        else:
            keys = [key]

        values = []
        for key in keys:
            t_df = df[df.Tag == key]
            value = -1
            for idx in sorted(set(t_df.Idx)):
                value = t_df[t_df.Idx == idx].Value.mean()
                value = np.nan_to_num(value)
                if not inner:
                    writer.add_scalar(f'{prefix}{key}/{idx}', value, i + 1)

            print(f'{prefix}{key}: {value:.5f}.')
            values.append(value)

            if inner:
                self.write_inner(df, key=key, i=i, writer=writer, prefix=prefix)

        return keys, values

    def write_inner(self, df, key, i, writer: Optional[SummaryWriter] = None, prefix=''):
        """

        Args:
            df: [Tag, Idx, Value]
            key: in Tag
            i:
            writer:
            prefix:

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        t_df = df[df.Tag == key]

        fig, ax = plt.subplots()
        ax.grid(True)
        sns.lineplot(t_df, x='Idx', y='Value', ax=ax)

        writer.add_figure(f"{prefix}{key}", fig, i + 1)

    def write_hv(self, mo_dict, i, ref=0, writer: Optional[SummaryWriter] = None, target='acc', norm=False,
                 prefix='hv'):
        """

        Args:
            mo_dict: dataframe ['Tag', 'Pop_id', 'Obj_id', 'Inner_id', 'Value']
            ref: ref for cal hv
            writer:
            target: also for mo_dict's Tag selector.
            norm: either to do normalization for n_inner > 1
            prefix:
            i: indicate x axis

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        t_df = mo_dict[mo_dict.Tag == target]
        n_pop = len(set(t_df.Pop_id))
        n_inner = len(set(t_df.Inner_id))
        n_obj = len(set(t_df.Obj_id))
        objs = np.array([[[
            t_df[(t_df.Pop_id == pop_idx) & (t_df.Obj_id == obj_idx) & (
                        t_df.Inner_id == inner_idx)].Value.mean()
            for pop_idx in range(n_pop)] for obj_idx in range(n_obj)] for inner_idx in range(n_inner)
        ])  # [n_inner, n_obj, n_pop]
        objs = np.nan_to_num(objs)

        '''for normalization'''
        if norm:
            min_objs = np.min(np.min(objs, axis=2, keepdims=True), axis=0, keepdims=True) - 1e-10
            max_objs = np.max(np.max(objs, axis=2, keepdims=True), axis=0, keepdims=True)
            objs = (objs - min_objs) / (max_objs - min_objs)

        '''cal hv for each inner mo'''
        hv = -1
        if ref == 'relative':
            ref = np.mean(objs[0], axis=-1).tolist()  # [n_obj]   to be list
        for inner_step in range(n_inner):
            hv = cal_hv(objs[inner_step], ref, target='min' if target == 'loss' else 'max')
            writer.add_scalar(f'{prefix}_details/{target}/{i+1}', hv, inner_step + 1)
        writer.add_scalar(f'{prefix}/{target}', hv, i + 1)

        print(f"==>> {prefix}: {target} {hv:.3f}.")

    def write_avg_span(self, mo_dict, i, writer: Optional[SummaryWriter] = None, target='acc', norm=False,
                       prefix='avg_span'):
        """
        E_i(max(f_i) - min(f_i))
        Args:
            mo_dict: dataframe ['Tag', 'Pop_id', 'Obj_id', 'Inner_id', 'Value']
            writer:
            target: also for mo_dict's Tag selector.
            norm: either to norm obj space (use for n_inner > 1)
            prefix:
            i: indicate x axis

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        t_df = mo_dict[mo_dict.Tag == target]
        n_pop = len(set(t_df.Pop_id))
        n_inner = len(set(t_df.Inner_id))
        n_obj = len(set(t_df.Obj_id))
        objs = np.array([[[
            t_df[(t_df.Pop_id == pop_idx) & (t_df.Obj_id == obj_idx) & (
                        t_df.Inner_id == inner_idx)].Value.mean()
            for pop_idx in range(n_pop)] for obj_idx in range(n_obj)] for inner_idx in range(n_inner)
        ])  # [n_inner, n_obj, n_pop]
        objs = np.nan_to_num(objs)

        '''for normalization'''
        if norm:
            min_objs = np.min(np.min(objs, axis=2, keepdims=True), axis=0, keepdims=True) - 1e-10
            max_objs = np.max(np.max(objs, axis=2, keepdims=True), axis=0, keepdims=True)
            objs = (objs - min_objs) / (max_objs - min_objs)

        '''cal avg span for each inner mo'''
        avg_span = -1
        for inner_step in range(n_inner):
            avg_span = np.mean(
                [np.max(objs[inner_step][obj_idx]) - np.min(objs[inner_step][obj_idx]) for obj_idx in
                 range(n_obj)])
            writer.add_scalar(f'{prefix}_details/{target}/{i+1}', avg_span, inner_step + 1)
        writer.add_scalar(f'{prefix}/{target}', avg_span, i + 1)

        print(f"==>> {prefix}: {target} {avg_span:.5f}.")

        return avg_span

    def write_min_crowding_distance(self, mo_dict, i, writer: Optional[SummaryWriter] = None, target='acc', norm=False,
                                    prefix='min_cd'):
        """
        only for nd solutions. if min cd is inf, use avg_span instead.
        Args:
            mo_dict: dataframe ['Tag', 'Pop_id', 'Obj_id', 'Inner_id', 'Value']
            writer:
            target: also for mo_dict's Tag selector.
            norm: either to norm obj space (use for n_inner > 1)
            prefix:
            i: indicate x axis

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        t_df = mo_dict[mo_dict.Tag == target]
        n_pop = len(set(t_df.Pop_id))
        n_inner = len(set(t_df.Inner_id))
        n_obj = len(set(t_df.Obj_id))
        objs = np.array([[[
            t_df[(t_df.Pop_id == pop_idx) & (t_df.Obj_id == obj_idx) & (
                        t_df.Inner_id == inner_idx)].Value.mean()
            for pop_idx in range(n_pop)] for obj_idx in range(n_obj)] for inner_idx in range(n_inner)
        ])  # [n_inner, n_obj, n_pop]
        objs = np.nan_to_num(objs)

        '''for normalization'''
        if norm:
            min_objs = np.min(np.min(objs, axis=2, keepdims=True), axis=0, keepdims=True) - 1e-10
            max_objs = np.max(np.max(objs, axis=2, keepdims=True), axis=0, keepdims=True)
            objs = (objs - min_objs) / (max_objs - min_objs)

        '''cal min crowding distance for each inner mo (after nd sort)'''
        cd = -1
        for inner_step in range(n_inner):
            cd = cal_min_crowding_distance(objs[inner_step])
            if cd == np.inf:
                avg_span = np.mean(
                    [np.max(objs[inner_step][obj_idx]) - np.min(objs[inner_step][obj_idx]) for obj_idx in
                     range(n_obj)])
                cd = avg_span
            writer.add_scalar(f'{prefix}_details/{target}/{i+1}', cd, inner_step + 1)
        writer.add_scalar(f'{prefix}/{target}', cd, i + 1)

        print(f"==>> {prefix}: {target} {cd:.5f}.")

        return cd

    def write_mo(self, mo_dict, pop_labels, i, writer: Optional[SummaryWriter] = None, target='acc',
                 prefix='mo'):
        """
        draw mo graph for different inner step.
        Args:
            mo_dict: {pop_idx: {inner_idx: [n_obj]}} or
                dataframe ['Tag', 'Pop_id', 'Obj_id', 'Epoch_id', 'Inner_id', 'Value']
            pop_labels:
            i:
            writer:
            target: for mo_dict's Tag selector.
            prefix:

        Returns:

        """
        level = self.levels.index('INFO')
        if level < self.level:
            return

        if type(mo_dict) is dict:
            n_pop, n_inner, n_obj = len(mo_dict), len(mo_dict[0]), len(mo_dict[0][0])
            objs = np.array([
                [[mo_dict[pop_idx][inner_idx][obj_idx] for pop_idx in range(n_pop)]
                 for obj_idx in range(n_obj)] for inner_idx in range(n_inner)
            ])  # [n_inner, n_obj, n_pop]

            '''log objs figure'''
            if n_obj == 2:
                fig, ax = plt.subplots(1, 1)
            else:       # n_obj > 2
                fig, ax = plt.subplots(1, 1, subplot_kw={'polar': True})
            draw_objs(objs, pop_labels, ax=ax)
            writer.add_figure(f"{prefix}/objs_{target}", fig, i + 1)

            with open(os.path.join(writer.log_dir, f'{prefix}_mo_dict_{target}.json'), 'w') as f:
                json.dump(mo_dict, f)
        else:
            # for exp in set(mo_dict.Exp):
            #     for inner_lr in set(mo_dict.Inner_lr):
            #         for logit_scale in set(mo_dict.Logit_scale):
            #             t_df = mo_dict[(mo_dict.Tag == target) &
            #                            (mo_dict.Exp == exp) & (mo_dict.Inner_lr == inner_lr) &
            #                            (mo_dict.Logit_scale == logit_scale)]
            t_df = mo_dict[mo_dict.Tag == target]
            n_pop = len(set(t_df.Pop_id))
            n_epoch = len(set(t_df.Epoch_id))
            n_inner = len(set(t_df.Inner_id))
            n_obj = len(set(t_df.Obj_id))
            objs = np.array([[[[
                t_df[(t_df.Pop_id == pop_idx) & (t_df.Obj_id == obj_idx) & (
                            t_df.Inner_id == inner_idx) & (t_df.Epoch_id == epoch_idx)].Value.iloc[0]    # .mean()
                for pop_idx in range(n_pop)] for obj_idx in range(n_obj)]
                for inner_idx in range(n_inner)] for epoch_idx in range(n_epoch)])
            # .mean() is average over epoch, thus, not correct mo. just visualize the first one in this epoch
            # [n_epoch, n_inner, n_obj, n_pop]
            objs = np.nan_to_num(objs)
            if len(objs.shape) != 2:        # no log
                return

            '''log objs figure along epoch for last inner step'''
            if n_obj == 2:
                fig, ax = plt.subplots(1, 1)
            else:       # n_obj > 2
                fig, ax = plt.subplots(1, 1, subplot_kw={'polar': True})
            draw_objs(objs[:, -1], pop_labels, ax=ax)      # [:,-1,:2,:] for the first 2 axes
            # writer.add_figure(f"objs_{target}_{exp}_innerlr_{inner_lr}{prefix}/logit_scale_{logit_scale}",
            #                   figure, i + 1)
            writer.add_figure(f"{prefix}/{target}_epoch", fig, i + 1)

            '''log objs figure along inner step for all epoch'''
            for e in range(n_epoch):
                if n_obj == 2:
                    fig, ax = plt.subplots(1, 1)
                else:       # n_obj > 2
                    fig, ax = plt.subplots(1, 1, subplot_kw={'polar': True})
                draw_objs(objs[e], pop_labels, ax=ax)   # [e, :, :2, :] for the first 2 axes
                writer.add_figure(f"{prefix}_inner/{target}_t{i+1}", fig, e)

    def write_task(self, pmo, task: dict, task_title, i, writer: Optional[SummaryWriter] = None, prefix='task'):
        """

        Args:
            pmo:
            task: ['context_images', 'target_images', 'context_labels', 'target_labels']
            task_title:
            i:
            writer:
            prefix:

        Returns:

        """
        level = self.levels.index('DEBUG')
        if level < self.level:
            return

        '''write images'''
        imgs = torch.cat([task['context_images'], task['target_images']]).cuda()
        numpy_imgs = imgs.cpu().numpy()
        writer.add_images(f"{prefix}/{task_title}", numpy_imgs, i+1)

        '''log img sim in the task'''
        with torch.no_grad():
            img_features = pmo.embed(imgs)
            _, selection_info = pmo.selector(img_features, gumbel=False, hard=False, average=False)
            img_sim = selection_info['y_soft']  # [img_size, 10]
            _, selection_info = pmo.selector(img_features, gumbel=False, hard=False)
            tsk_sim = selection_info['y_soft']  # [1, 10]
        sim = torch.cat([img_sim, *[tsk_sim] * (img_sim.shape[0] // 10)]).cpu().numpy()
        figure = draw_heatmap(sim, verbose=False)
        writer.add_figure(f"{prefix}/{task_title}/sim", figure, i + 1)


def unnormalize(sample):
    # 标准化图像的逆变换
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    unnormalize = transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)],
                                       std=[1 / s for s in std])

    # 将图像标准化到0到1范围
    normalized_image = unnormalize(sample)
    normalized_image = torch.clamp(normalized_image, 0, 1)

    # 转换为Numpy数组
    numpy_image = normalized_image.permute(1, 2, 0).cpu().numpy()

    # 缩放像素值到0到255范围
    numpy_image = (numpy_image * 255).astype(np.uint8)

    image = Image.fromarray(numpy_image, mode='RGB')

    return image


def visualize_att_map(att_map, image, grid_size=14, alpha=0.6, ax=None):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    assert len(att_map.shape) == 1

    mask = att_map.reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 7))
        fig.tight_layout()

    ax.imshow(image)
    ax.imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')
    # ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_att_map_with_color(att_map, image, color, grid_size=14, alpha=0.6, ax=None):
    # color with range [0-1, 0-1, 0-1]
    # argmax att
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    assert len(att_map.shape) == 1

    # mask attn to 0,1
    mask = att_map.reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size), Image.BICUBIC)
    mask = np.asarray(mask)
    mask = mask - mask.min()
    mask = mask / mask.max()

    colored_mask = np.stack([np.zeros_like(mask) for _ in range(4)], axis=2)
    colored_mask[mask > 0.5] = [*color, alpha]
    # alpha_mask = np.ones_like(mask, dtype=float)
    # alpha_mask[mask > 0.5] = alpha
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 7))
        fig.tight_layout()

    # ax.imshow(ori_x)
    ax.imshow(colored_mask)  # , alpha=alpha_mask
    # ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
