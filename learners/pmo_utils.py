from typing import List, Dict, Any, Optional
import os
import shutil
import json
import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from pymoo.visualization.pcp import PCP

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from mo_optimizers import hv_maximization


class Pool(data.Dataset):     # (nn.Module)
    """
    Pool stored class samples.

    A class instance contains (a set of image samples, class_label, class_label_str).
    """
    def __init__(self, memory_size=1000, seed=0):
        """
        :param memory_size: Maximum number of images can be stored in the pool.
        """
        self.memory_size = memory_size
        self.seed = seed
        self.clusters = []

    def __len__(self):      # all images
        if len(self.clusters) == 0:
            return 0

        length = np.sum([len(cls['images']) for task in self.clusters for cls in task])
        return length

    def length(self, taskid=None, classid=None):
        """
        return a or a list of num_imgs according to taskid and classid
        """
        if taskid is None and classid is None:
            return [[len(cls['images']) for cls in task] for task in self.clusters]
        if classid is None:
            return [len(cls['images']) for cls in self.clusters[taskid]]
        else:
            return len(self.clusters[taskid][classid])

    def num_clusters(self):
        return len(self.clusters)

    def __getitem__(self, item):

        len_list = self.length()
        # locate task
        integral_tsk = np.array([sum(sum(len_list[:taskid+1], [])) for taskid in range(len(len_list))])
        # since len_list is 2-D and may have diff len for some tasks, use sum(list, []) to flatten array.
        task_idx = sum(integral_tsk < (item + 1))
        item_offset = 0 if task_idx == 0 else integral_tsk[task_idx - 1]

        # locate class
        integral_cls = np.array([np.sum(
            len_list[task_idx][:classid+1]
        ) for classid in range(len(len_list[task_idx]))])
        class_idx = sum(integral_cls < (item + 1 - item_offset))
        item_offset += 0 if class_idx == 0 else integral_cls[class_idx - 1]

        # locate image
        image_idx = item - item_offset

        return (torch.from_numpy(self.clusters[task_idx][class_idx]['images'][image_idx]),
                self.clusters[task_idx][class_idx]['label'],
                task_idx)

    def return_random_dataset(self, size):
        """Return a dataset with num_samples = size, randomly sampled from the pool"""
        class PoolRandomDataset(data.Dataset):
            def __init__(self, _pool, _size):
                self.pool = _pool
                self.pool_size = len(_pool)
                self.size = _size
                self.rng = np.random.RandomState(_pool.seed + 1)
                self.items = self.rng.choice(self.pool_size, self.size)
                # mapping of item to exact item in pool
                # size can > pool_size

                self.images = []
                self.targets = []
                self.tasks = []
                for item in self.items:
                    self.images.append(self.pool[item][0])
                    self.targets.append(self.pool[item][1])
                    self.tasks.append(self.pool[item][2])
                self.images = torch.stack(self.images)
                self.targets = np.array(self.targets)
                self.tasks = np.array(self.tasks)

            def __len__(self):
                return self.size

            def __getitem__(self, item):
                return self.images[item], self.targets[item], self.tasks[item]
                # return self.pool[self.items[item]]

        return PoolRandomDataset(self, size)

    def clear(self):
        clusters = self.clusters
        self.clusters = []
        return clusters

    def put(self, images, info_dict):
        """ Put a task
        Put samples (batch of torch cpu or numpy images) into clusters.
            info_dict should contain `labels`, `tasks`,     # numpy
        In this work, samples should be all samples for one task.
        """
        '''unpack'''
        labels, tasks = info_dict['labels'], info_dict['tasks']
        # similarities, features = info_dict['similarities'], info_dict['features']
        for task in np.sort(np.unique(tasks, axis=0)):
            # make up self.clusters
            while task >= self.num_clusters():
                self.clusters.append([])

            task_mask = tasks == task
            for label in np.unique(labels[task_mask], axis=0):     # unique along first axis
                mask = (labels == label) & (tasks == task)     # same task and same label
                class_images = images[mask].numpy() if type(images) == torch.Tensor else images[mask]
                # class_features = features[mask]
                # class_similarities = similarities[mask]

                '''put cls to cluster[task]'''
                cls = {'images': class_images, 'label': label,}
                if len(self.clusters[task]) > 0:    # has classes
                    ls = np.array([c['label'] for c in self.clusters[task]])
                    c_idxs = np.where(ls == label)[0]
                    if len(c_idxs) == 0:    # no stored label
                        self.clusters[task].append(cls)
                    elif len(c_idxs) == 1:  # already stored label
                        stored = self.clusters[task][c_idxs[0]]['images']
                        self.clusters[task][c_idxs[0]]['images'] = np.concatenate([stored, class_images])
                    else:       # > 1
                        raise Exception(f'Multiple same classes in task{task}: label{label}.')

                else:
                    self.clusters[task].append(cls)

        '''cal num_samples_each_class for the updated clusters'''
        num_classes = np.sum([len(task) for task in self.clusters])
        num_samples_each_class = int(self.memory_size // num_classes)

        '''balance num of samples in each class'''
        # coreset selection without affecting RNG state
        state = np.random.get_state()
        np.random.seed(self.seed)     #
        for task in self.clusters:
            for cls in task:
                img_len = len(cls['images'])
                remain_len = int(min(num_samples_each_class, img_len))
                cls['images'] = cls['images'][np.random.choice(img_len, remain_len, replace=False)]
        np.random.set_state(state)

    def find_label(self, label: np.ndarray, cluster_idx=None):
        """
        Find label in pool, return position with (cluster_idx, cls_idx)
        If not in pool, return -1.
        If target == 'buffer', return the position (idx) in the buffer.
        """
        if cluster_idx is not None:
            for cls_idx, cls in enumerate(self.clusters[cluster_idx]):
                if (cls['label'] == label).all():       # (0, str) == (0, str) ? int
                    return cluster_idx, cls_idx
        else:
            for cluster_idx, cluster in enumerate(self.clusters):
                for cls_idx, cls in enumerate(cluster):
                    if (cls['label'] == label).all():       # (0, str) == (0, str) ? int
                        return cluster_idx, cls_idx

        return -1

    def current_clusters(self):
        """
        Return clusters: (cluster_name, cluster)
        """
        clusters = []
        for cluster_name, cluster in enumerate(self.clusters):
            clusters.append((cluster_name, cluster))

        return clusters

    def current_classes(self):
        """
        Return current classes stored in the pool (name, num_images)
        """
        clses = []
        for cluster in self.clusters:
            clses_in_cluster = []
            for cls in cluster:
                clses_in_cluster.append((cls['label'], cls['images'].shape[0]))
            clses.append(clses_in_cluster)
        return clses

    def current_images(self, single_image=False):
        """
        # Return a batch of images (torch.Tensor) in the current pool with pool_montage.
        # batch of images => (10, 3, 84, 84)
        # class_montage => (3, 84, 84*10)
        # cluster montage => (3, 84*max_num_classes, 84*10)
        # pool montage => (3, 84*max_num_classes, 84*10*capacity).

        first return raw list, [8 * [num_class_each_cluster * numpy [10, 3, 84, 84]]]
        with labels
        """
        images = []
        for cluster in self.clusters:
            imgs = []
            for cls in cluster:
                imgs.append(cls['images'])      # cls['images'] shape [10, 3, 84, 84]
            images.append(imgs)

        if single_image:
            '''obtain width of images'''
            # max_num_imgs = self.max_num_images
            max_num_imgs = 0
            for cluster_idx, cluster in enumerate(images):
                for cls_idx, cls in enumerate(cluster):
                    num_imgs = cls.shape[0]
                    if num_imgs > max_num_imgs:
                        max_num_imgs = num_imgs

            '''construct a single image for each cluster'''
            for cluster_idx, cluster in enumerate(images):
                for cls_idx, cls in enumerate(cluster):
                    imgs = np.zeros((max_num_imgs, *cls.shape[1:]))
                    if len(cls) > 0:    # contain images
                        imgs[:cls.shape[0]] = cls
                    cluster[cls_idx] = np.concatenate([
                        imgs[img_idx] for img_idx in range(max_num_imgs)], axis=-1)
                if len(cluster) > 0:
                    images[cluster_idx] = np.concatenate(cluster, axis=-2)
                    # [3, 84*num_class, 84*max_num_images_in_class]
                    # [3, 84*50, 84*20]
                # else:   # empty cluster
                #     images[cluster_idx] = np.zeros((3, 84, 84))

        return images

    def episodic_sample(
            self,
            cluster_idx,
            n_way,
            n_shot,
            n_query,
            remove_sampled_classes=False,
            d='numpy',
    ):
        """
        Sample a task from the specific cluster_idx.
        length of this cluster needs to be guaranteed larger than n_way.
        Random issue may occur, highly recommended to use np.rng.
        Return numpy if d is `numpy`, else tensor on d
        """
        candidate_class_idxs = np.arange(len(self.clusters[cluster_idx]))
        num_imgs = np.array([cls[1] for cls in self.current_classes()[cluster_idx]])
        candidate_class_idxs = candidate_class_idxs[num_imgs >= n_shot + n_query]
        assert len(candidate_class_idxs) >= n_way

        selected_class_idxs = np.random.choice(candidate_class_idxs, n_way, replace=False)
        context_images, target_images, context_labels, target_labels, context_gt, target_gt = [], [], [], [], [], []
        # context_features, target_features = [], []
        # context_selection, target_selection = [], []
        for re_idx, idx in enumerate(selected_class_idxs):
            images = self.clusters[cluster_idx][idx]['images']              # [bs, c, h, w]
            tuple_label = self.clusters[cluster_idx][idx]['label']          # (gt_label, domain)
            # features = self.clusters[cluster_idx][idx]['features']
            # selection = self.clusters[cluster_idx][idx]['selection']        # [bs, n_clusters]

            perm_idxs = np.random.permutation(np.arange(len(images)))
            context_images.append(images[perm_idxs[:n_shot]])
            target_images.append(images[perm_idxs[n_shot:n_shot+n_query]])
            # context_features.append(features[perm_idxs[:n_shot]])
            # target_features.append(features[perm_idxs[n_shot:n_shot+n_query]])
            context_labels.append([re_idx for _ in range(n_shot)])
            target_labels.append([re_idx for _ in range(n_query)])
            context_gt.append([tuple_label for _ in range(n_shot)])         # [(gt_label, domain)*n_shot]
            target_gt.append([tuple_label for _ in range(n_query)])         # [(gt_label, domain)*n_query]
            # context_selection.append(selection[perm_idxs[:n_shot]])
            # target_selection.append(selection[perm_idxs[n_shot:n_shot+n_query]])

        context_images = np.concatenate(context_images)
        target_images = np.concatenate(target_images)
        # context_features = np.concatenate(context_features)
        # target_features = np.concatenate(target_features)
        context_labels = np.concatenate(context_labels)
        target_labels = np.concatenate(target_labels)
        context_gt = np.concatenate(context_gt)
        target_gt = np.concatenate(target_gt)
        # context_selection = torch.cat(context_selection)
        # target_selection = torch.cat(target_selection)

        '''to tensor on divice d'''
        if d != 'numpy':
            context_images = torch.from_numpy(context_images).to(d)
            target_images = torch.from_numpy(target_images).to(d)
            # context_features = torch.from_numpy(context_features).to(d)
            # target_features = torch.from_numpy(target_features).to(d)
            context_labels = torch.from_numpy(context_labels).long().to(d)
            target_labels = torch.from_numpy(target_labels).long().to(d)

        task_dict = {
            'context_images': context_images,           # shape [n_shot*n_way, 3, 84, 84]
            'context_labels': context_labels,           # shape [n_shot*n_way,]
            'context_gt': context_gt,                   # shape [n_shot*n_way, 2]: [local, domain]
            'target_images': target_images,             # shape [n_query*n_way, 3, 84, 84]
            'target_labels': target_labels,             # shape [n_query*n_way,]
            'target_gt': target_gt,                     # shape [n_query*n_way, 2]: [local, domain]
            'domain': cluster_idx,                      # 0-7: C0-C7, num_clusters
            # 'context_features': context_features,       # shape [n_shot*n_way, 512]
            # 'target_features': target_features,         # shape [n_query*n_way, 512]
            # 'context_selection': context_selection,     # shape [n_shot*n_way, n_clusters]
            # 'target_selection': target_selection,       # shape [n_query*n_way, n_clusters]
        }

        if remove_sampled_classes:
            class_items = []
            for idx in range(len(self.clusters[cluster_idx])):
                if idx not in selected_class_idxs:
                    class_items.append(self.clusters[cluster_idx][idx])
            self.clusters[cluster_idx] = class_items

        return task_dict

class Mixer:
    """
    Mixer used to generate mixed tasks.
    """
    def __init__(self, mode='cutmix', num_sources=2, num_mixes=2):
        """
        :param mode indicates how to generate mixed tasks.
        :param num_sources indicates how many tasks are used to generate 1 mixed task.
        :param num_mixes indicates how many mixed tasks needed to be generated.
        """
        self.mode = mode
        if mode not in ['cutmix']:      # 'mixup'
            raise Exception(f'Un implemented mixer mode: {mode}.')
        self.num_sources = num_sources
        self.num_mixes = num_mixes
        self.ref = get_reference_directions("energy", num_sources, num_sources+num_mixes, seed=1234)

        '''eliminate num_obj extreme cases.'''
        check = np.sum(self.ref == 1, axis=1) == 0             # [[0,0,1]] == 1 => [[False, False, True]]
        # np.sum(weights == 1, axis=1): array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        self.ref = self.ref[check]      # shape [num_mixes, num_sources]    e.g. [[0.334, 0.666], [0.666, 0.334]]
        # self.ref = get_reference_directions("energy", num_obj, num_mix, seed=1)  # use those [1, 0, 0]
        assert self.ref.shape[0] == num_mixes

    def _cutmix(self, task_list, mix_id):
        """
        Apply cutmix on the task_list.
        task_list contains a list of task_dicts:
            {context_images, context_labels, context_gt, target_images, target_labels, target_gt, domain,
             context_selection, target_selection}
        mix_id is used to identify which ref to use as a probability.

        task sources should have same size, so that the mixed image is corresponding to the same position in sources.

        return:
        task_dict = {
            'context_images': context_images,           # shape [n_shot*n_way, 3, 84, 84]
            'context_features': context_features,       # shape [n_shot*n_way, 512]
            'context_labels': context_labels,           # shape [n_shot*n_way,]
            'target_images': target_images,             # shape [n_query*n_way, 3, 84, 84]
            'target_features': target_features,         # shape [n_query*n_way, 512]
            'target_labels': target_labels,             # shape [n_query*n_way,]
            }
        meta_info: {'probability': probability of chosen which background,
         'lam': the chosen background for each image [(n_shot+n_query)* n_way,]}
        """
        # identify image size
        _, c, h, w = task_list[0]['context_images'].shape
        # _, fs = task_list[0]['context_features'].shape
        context_size_list = [task_list[idx]['context_images'].shape[0] for idx in range(len(task_list))]
        target_size_list = [task_list[idx]['target_images'].shape[0] for idx in range(len(task_list))]
        assert np.min(context_size_list) == np.max(context_size_list)   # assert all contexts have same size
        assert np.min(target_size_list) == np.max(target_size_list)     # assert all targets have same size
        context_size = np.min(context_size_list)
        target_size = np.min(target_size_list)
        # print(f'context_size: {context_size}, target_size: {target_size}.')

        # generate num_sources masks for imgs with size [c, h, w]
        cutmix_prop = 0.3   # for (84*84), cut region is int(84*0.3)= (25*25)
        cuth, cutw = int(h * cutmix_prop), int(w * cutmix_prop)  # 84*0.3 [25, 25]
        # cutfs = int(fs * cutmix_prop)  # 84*0.3 [25, 25]

        # generate lam, which is the index of img to be background. other imgs are foreground.
        # based on weight as probability.
        probability = self.ref[mix_id]      # shape [num_sources, ], sum = 1
        lam = np.random.choice(self.num_sources, context_size+target_size, p=probability, replace=True)
        # lam with shape [context_size+target_size,] is the decision to use which source as background.

        mix_imgs = []   # mix images batch
        # mix_feas = []   # mix features batch
        mix_labs = []   # mix relative labels batch, same [0,0,1,1,2,2,...]
        # mix_gtls = []   # mix gt labels batch, str((weighted local label, domain=-1))
        for idx in range(context_size+target_size):
            if idx < context_size:
                set_name = 'context_images'
                # set_nafs = 'context_features'
                lab_name = 'context_labels'
            else:
                set_name = 'target_images'
                # set_nafs = 'target_features'
                lab_name = 'target_labels'
            # gtl_name = 'context_gt' if img_idx < context_size else 'target_gt'

            img_idx = idx if idx < context_size else idx - context_size     # local img idx in context and target set.
            # mix img is first cloned with background.
            mix_img = task_list[lam[idx]][set_name][img_idx].copy()
            # mix_fea = task_list[lam[idx]][set_nafs][img_idx].copy()

            # for other foreground, cut the specific [posihs: posihs+cuth, posiws: posiws+cutw] region to
            # mix_img's [posiht: posiht+cuth, posiwt: posiwt+cutw] region
            for fore_img_idx in np.delete(np.arange(self.num_sources), lam[idx]):  # idxs for other imgs
                # pick pixels from [posihs, posiws, cuth, cutw], then paste to [posiht, posiwt, cuth, cutw]
                posihs = np.random.randint(h - cuth)
                posiws = np.random.randint(w - cutw)
                posiht = np.random.randint(h - cuth)
                posiwt = np.random.randint(w - cutw)
                # posifss = np.random.randint(fs - cutfs)
                # posifst = np.random.randint(fs - cutfs)

                fore = task_list[fore_img_idx][set_name][img_idx][:, posihs: posihs + cuth, posiws: posiws + cutw]
                mix_img[:, posiht: posiht + cuth, posiwt: posiwt + cutw] = fore
                mix_imgs.append(mix_img)
                # fofs = task_list[fore_img_idx][set_nafs][img_idx][posifss: posifss + cutfs]
                # mix_fea[posifst: posifst + cutfs] = fofs
                # mix_feas.append(mix_fea)

            # determine mix_lab  same as the chosen img
            mix_labs.append(task_list[lam[idx]][lab_name][img_idx])

            # # determine mix_gtl  str((weighted local label, domain=-1))
            # local_label = np.sum(
            #     probability * np.array([task_list[idx][gtl_name][img_idx][0] for idx in range(self.num_sources)]))
            # domain = -1
            # mix_gtls.append('({:.2f}, {})'.format(local_label, domain))

        # formulate to task
        task_dict = {
            'context_images': np.stack(mix_imgs[:context_size]),   # shape [n_shot*n_way, 3, 84, 84]
            # 'context_features': np.stack(mix_feas[:context_size]),       # shape [n_shot*n_way, 512]
            'context_labels': np.array(mix_labs[:context_size]),   # shape [n_shot*n_way,]
            'target_images': np.stack(mix_imgs[context_size:]),     # shape [n_query*n_way, 3, 84, 84]
            # 'target_features': np.stack(mix_feas[context_size:]),         # shape [n_query*n_way, 512]
            'target_labels': np.array(mix_labs[context_size:]),     # shape [n_query*n_way,]
        }

        return task_dict, {'probability': probability, 'lam': lam}

    def mix(self, task_list, mix_id=0):
        """
        Numpy task task_list, len(task_list) should be same with self.num_sources
        """
        assert len(task_list) == self.num_sources
        assert mix_id < self.num_mixes
        assert isinstance(task_list[0]['context_images'], np.ndarray)

        if self.mode == 'cutmix':
            return self._cutmix(task_list, mix_id)

    def visualization(self, task_list):
        """
        Visualize the generated mixed task for all num_mixes cases.
        """
        pass


def compute_prototypes(embeddings, labels, n_way):
    prots = torch.zeros(n_way, embeddings.shape[-1]).type(
        embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        if torch.__version__.startswith('1.1'):
            prots[i] = embeddings[(labels == i).nonzero(), :].mean(0)
        else:
            prots[i] = embeddings[(labels == i).nonzero(as_tuple=False), :].mean(0)
    return prots


# return cos similarities for each prot (class_centroid) to cluster_centers
def prototype_similarity(embeddings, labels, centers, distance='cos'):
    """
    :param embeddings: shape [task_img_size, emb_dim]
    :param labels: relative labels shape [task_img_size,], e.g., [0, 0, 0, 1, 1, 1]
    :param centers: shape [n_clusters, emb_dim]
    :param distance: similarity [cos, l2, lin, corr]

    :return similarities shape [n_way, n_clusters] and class_centroids shape [n_way, emb_dim]
    """
    n_way = len(labels.unique())

    class_centroids = compute_prototypes(embeddings, labels, n_way)
    prots = class_centroids.unsqueeze(1)      # shape [n_way, 1, emb_dim]
    centers = centers.unsqueeze(0)         # shape [1, n_clusters, emb_dim]

    if distance == 'l2':
        logits = -torch.pow(centers - prots, 2).sum(-1)    # shape [n_way, n_clusters]
    elif distance == 'cos':
        logits = F.cosine_similarity(centers, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        logits = torch.einsum('izd,zjd->ij', prots, centers)
    elif distance == 'corr':
        logits = F.normalize((centers * prots).sum(-1), dim=-1, p=2) * 10
    else:
        raise Exception(f"Un-implemented distance {distance}.")

    return logits, class_centroids


def cal_hv_weights(objs, ref=None, reverse=False):
    """
    HV loss calculation: weighted loss
    code function from HV maximization:
    https://github.com/timodeist/multi_objective_learning/tree/06217d0ce024b92d52cdeb0390b1afb29ee59819

    Args:
        objs: Tensor/ndarry with shape(obj_size, pop_size)     e.g., (3, 6)
        ref:
        reverse: True if use negative objs and return negative weights for loss maximization. False otherwise.

    Returns:
        weighted loss, for which the weights are based on HV gradients.
    """

    num_obj, num_sol = objs.shape[0], objs.shape[1]
    if ref is not None:
        ref = np.array([ref for _ in range(num_obj)])

    # obtain weights for the points in this front
    mo_opt = hv_maximization.HvMaximization(num_sol, num_obj, ref)

    # obtain np objs
    if type(objs) is torch.Tensor:
        objs_np = objs.detach().cpu().numpy()
    else:
        objs_np = objs

    # compute weight for each solution
    if reverse:
        objs_np = -objs_np
    weights = mo_opt.compute_weights(objs_np)       # check if use objs_np.transpose()
    if reverse:
        weights = -weights

    if type(objs) is torch.Tensor:
        weights = weights.to(objs.device)
    #     # weights = weights.permute([1, 0]).to(objs.device)
    #     weighted_loss = torch.sum(objs * weights, dim=0)
    # else:
    #     weights = weights.numpy()
    #     # weights = weights.permute([1, 0]).numpy()
    #     weighted_loss = np.sum(objs * weights, axis=0)

    return weights


def normalize(tensor, dim=0, p=1/2, noise=False):
    # proj to simplex: p=1 sum(f)=1, sphere: p=2 sum(f^2)^(1/2)=1, concave-sphere: p=1/2 sum(f^(1/2))^2=1
    tensor = torch.clamp(tensor, min=1e-5)  # clamp negative value and too small value

    if noise:       # prevent the same point
        noise = torch.from_numpy(np.random.rand(*tensor.shape)).float().to(tensor.device) * 2e-6 - 1e-6
        # noise = torch.rand_like(tensor) * 2e-9 - 1e-9     # [0, 1e-9]
        tensor = tensor + noise

    normalized_tensor = F.normalize(tensor, dim=dim, p=p, eps=1e-10)
    # sum_tensor = torch.sum(tensor ** p, dim=dim, keepdim=True) ** (1/p)
    # normalized_tensor = tensor / (sum_tensor + 1e-10)

    return normalized_tensor


def cal_hv(objs, ref=2, target='min'):
    """
    Calculate HV value for multi-objective losses and accs.

    Args:
        objs : Tensor/ndarry with shape(obj_size, pop_size)     e.g., (3, 6)
        ref: 2 for loss, 0 for acc
        target:

    Returns:
        hv value
    """

    num_obj, num_sol = objs.shape[0], objs.shape[1]
    if type(ref) is not list:
        ref_point = np.array([ref for _ in range(num_obj)])
    else:
        ref_point = np.array(ref)

    assert len(ref_point.shape) == 1

    # obtain np objs
    if type(objs) is torch.Tensor:
        objs_np = objs.detach().cpu().numpy()
    else:
        objs_np = objs

    # for acc reverse objs
    if target == 'max':
        objs_np = -objs_np

    ind = HV(ref_point=ref_point)
    hv = ind(objs_np.T)

    if type(hv) is not float:
        hv = hv.item()

    return hv


def cal_min_crowding_distance(objs):
    """
    code from pymoo, remove normalization part, return the min cd.
    Args:
        objs: Tensor/ndarry with shape(obj_size, pop_size)     e.g., (3, 6)

    Returns:

    """
    # obtain np objs: F
    if type(objs) is torch.Tensor:
        F = objs.detach().cpu().numpy().T
    else:
        F = objs.T

    non_dom = NonDominatedSorting().do(F, only_non_dominated_front=True)
    F = np.copy(F[non_dom, :])

    n_points, n_obj = F.shape

    # sort each column and get index
    I = np.argsort(F, axis=0, kind='mergesort')

    # sort the objective space values for the whole matrix
    F = F[I, np.arange(n_obj)]

    # calculate the distance from each point to the last and next
    dist = np.row_stack([F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), F])

    # # calculate the norm for each objective - set to NaN if all values are equal
    # norm = np.max(F, axis=0) - np.min(F, axis=0)
    # norm[norm == 0] = np.nan

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist, np.copy(dist)
    # dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm
    dist_to_last, dist_to_next = dist_to_last[:-1], dist_to_next[1:]

    # if we divide by zero because all values in one columns are equal replace by none
    dist_to_last[np.isnan(dist_to_last)] = 0.0
    dist_to_next[np.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = np.argsort(I, axis=0)
    cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    return min(cd)


def draw_objs(objs, labels=None, ax=None, legend=False, ax_labels=None):
    """
    Example fig: fig, ax = plt.subplots(1, 1, subplot_kw={'polar': True}, figsize=(10, 10))
    objs: numpy with shape [obj_size, pop_size] or [n_iter, obj_size, pop_size] with gradient color
    labels: list of labels: ['p0', 'p1', 'm0', 'm1'] or label str list for all pop
    ax_labels: label for axes, None to mute, otherwise specify
    """
    fig = None
    n_iter = 1
    if len(objs.shape) == 2:
        obj_size, pop_size = objs.shape
        objs = objs[np.newaxis, :, :]
    else:
        n_iter, obj_size, pop_size = objs.shape

    # assert obj_size == 2

    if obj_size == 2:
        '''generate pandas DataFrame for objs'''
        if labels is not None:
            data = pd.DataFrame({       # for all points
                'f1': [objs[i_idx, 0, pop_idx] for i_idx in range(n_iter) for pop_idx in range(pop_size)],
                'f2': [objs[i_idx, 1, pop_idx] for i_idx in range(n_iter) for pop_idx in range(pop_size)],
                'Iter': [i_idx for i_idx in range(n_iter) for pop_idx in range(pop_size)],
                'Label': [labels[pop_idx] for i_idx in range(n_iter) for pop_idx in range(pop_size)],
            })
        else:
            data = pd.DataFrame({       # for all points
                'f1': [objs[i_idx, 0, pop_idx] for i_idx in range(n_iter) for pop_idx in range(pop_size)],
                'f2': [objs[i_idx, 1, pop_idx] for i_idx in range(n_iter) for pop_idx in range(pop_size)],
                'Iter': [i_idx for i_idx in range(n_iter) for pop_idx in range(pop_size)],
            })

        if ax is None:
            fig, ax = plt.subplots()
        ax.grid(True)
        # c = plt.get_cmap('rainbow', pop_size)

        if labels is not None:
            sns.scatterplot(data, x='f1', y='f2',
                            hue='Label', size='Iter', sizes=(100, 200), alpha=1., ax=ax)
        else:
            sns.scatterplot(data, x='f1', y='f2',
                            size='Iter', sizes=(100, 200), alpha=1., ax=ax)

        if legend:
            # ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.1), ncol=1)
            ax.legend(loc='lower center', frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.2))
        else:
            ax.legend([], [], frameon=False)
    else:
        '''obj size larger than 2'''
        '''ax should with projection='polar' or polar=True '''
        if labels is not None:
            # 分配颜色 by to different iter
            assert n_iter == 1, f'only 1 iter can assign labels'
            objs = objs[0]      # [obj_size, pop_size]
            reshaped_objs = []
            label_map = {}
            label_int = []
            for label_i, label in enumerate(set(labels)):
                label_map[label_i] = label
                label_map[label] = label_i
            for label in labels:
                label_int.append(label_map[label])
            label_int = np.array(label_int)
            for label_i, label in enumerate(set(labels)):
                int_label = label_map[label]
                reshaped_objs.append(objs[:, label_int == int_label])

            # # remove some samples to match the same size
            # s = int(np.min([samples.shape[-1] for samples in reshaped_objs]))
            # reshaped_objs = [samples[:, :s] for samples in reshaped_objs]
            # reshaped_objs = np.stack(reshaped_objs)     # [n_label, obj_size, pop_size]
            objs = reshaped_objs        # n_label * [obj_size, pop_size(different)]
            # n_iter, obj_size, pop_size = objs.shape     # update pop size == s
            n_iter = len(objs)
            pop_size = None     # differ for different class(iter)

        if ax_labels is None:
            ax_labels = np.array([None for i in range(obj_size)])
        elif type(ax_labels) is str and ax_labels == 'default':
            ax_labels = np.array([r'$f_{{{}}}$'.format(i) for i in range(obj_size)])
        angles = np.linspace(0, 2 * np.pi, obj_size, endpoint=False)
        ax.set_thetagrids(angles * 180 / np.pi, ax_labels, fontsize=12)
        angles = np.concatenate([angles, [angles[0]]])

        num_draw_iter = 10
        color = sns.color_palette('tab10', num_draw_iter if n_iter > num_draw_iter else n_iter)
        for r_idx, i_idx in enumerate(np.linspace(
                0, n_iter-1, num_draw_iter if n_iter > num_draw_iter else n_iter, dtype=int)):
            pop_size = objs[i_idx].shape[-1]
            _objs = np.concatenate([objs[i_idx], objs[i_idx][0:1, :]], axis=0)  # cat last obj with first obj
            for sample_idx in range(pop_size):
                if sample_idx == 0 and labels is not None:     # first sample with legend
                    ex_params = {'label': label_map[i_idx]}
                else:
                    ex_params = {}
                ax.plot(angles, _objs[:, sample_idx], '-', linewidth=2 if labels is not None else (i_idx+1)/n_iter*5,
                        color=color[r_idx], alpha=0.6, **ex_params)

        ax.set_theta_zero_location('N')
        ax.set_rlabel_position(0)
        ax.tick_params(axis='both', labelsize=12)
        if legend:
            ax.legend(loc='lower center', frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.2))

        # pcp = PCP()     # legend=(True, {'loc': "upper left"}), cmap='Reds' x
        # pcp.set_axis_style(color="grey", alpha=0.5)
        # for i_idx in range(n_iter):
        #     pcp.add(np.transpose(objs[i_idx], (1, 0)), linewidth=(i_idx+1)/n_iter*5, label=f'{i_idx}')
        # fig = pcp.show().fig
        # plt.close(fig)

    return fig

def draw_heatmap(data, verbose=True, ax=None, fmt=".2f", cbar=True):
    """
    return a figure of heatmap.
    :param data: 2-D Numpy
    :param verbose: whether to use annot
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    if verbose:
        sns.heatmap(data, cmap=plt.get_cmap('Greens'), annot=True, fmt=fmt, cbar=False, ax=ax)
    else:
        sns.heatmap(data, cmap=plt.get_cmap('Greens'), cbar=cbar, ax=ax)
    return fig


def map_re_label(re_labels):
    """
    For masked re_label, it can be [1,1,1,3,3,3,7,7,7]
    As a valid episodic task, re_label should be in the form [0,0,0,1,1,1,2,2,2]
    Return valid re_label
    """
    label_set = np.unique(re_labels)
    re_label_map = {
        origin_label: correct_label for origin_label, correct_label in zip(label_set, np.arange(len(label_set)))}
    correct_labels = np.array(list(map(lambda x: re_label_map[x], re_labels)))
    return correct_labels


def available_setting(num_imgs_clusters, task_type, min_available_clusters=1, use_max_shot=False,
                      must_include_clusters=None):
    """Check whether pool has enough samples for specific task_type and return a valid setting.
    :param num_imgs_clusters: list of Numpy array with shape [num_clusters * [num_classes]]
                              indicating number of images for specific class in specific clusters.
    :param task_type: `standard`: vary-way-vary-shot-ten-query (maximum all 10)
                      `1shot`: five-way-one-shot-ten-query
                      `2shot`: five-way-two-shot-two-query
                      `5shot`: vary-way-five-shot-ten-query
    :param min_available_clusters: minimum number of available clusters to apply that setting.
    :param use_max_shot: if True, return max_shot rather than random shot
    :param must_include_clusters: nparray of cluster idxs to be included.
        If specified, only explore the chosen clusters.
        Default to None
    :return a valid setting.
    """
    n_way, n_shot, n_query = -1, -1, -1
    for _ in range(10):     # try 10 times, if still not available setting, return -1
        n_query = 2 if task_type == '2shot' else 10

        min_shot = 5 if task_type == '5shot' else 1
        min_way = 5
        # must include
        if must_include_clusters is not None:
            max_way = min([len(num_imgs_clusters[cluster_id][num_imgs_clusters[cluster_id] >= min_shot + n_query])
                           for cluster_id in must_include_clusters])
            min_available_clusters = len(must_include_clusters)
        else:
            # if not specified must_include_clusters, consider all clusters
            max_way = sorted(
                [len(num_images[num_images >= min_shot + n_query]) for num_images in num_imgs_clusters]
            )[::-1][min_available_clusters - 1]
            must_include_clusters = np.arange(len(num_imgs_clusters))

        max_way = int(min(10, max_way))      # too many ways cause CUDA out of memory

        if max_way < min_way:
            return -1, -1, -1   # do not satisfy the minimum requirement.

        n_way = 5 if task_type in ['1shot', '2shot'] else np.random.randint(min_way, max_way + 1)

        # shot depends on chosen n_way
        available_shots = []
        for cluster_id in must_include_clusters:
            num_images = num_imgs_clusters[cluster_id]
            shots = sorted(num_images[num_images >= min_shot + n_query])[::-1][:n_way]
            available_shots.append(0 if len(shots) < n_way else (shots[-1] - n_query))
        max_shot = np.min(sorted(available_shots)[::-1][:min_available_clusters])

        max_shot = int(min(10, max_shot))       # too many shots cause CUDA out of memory

        if max_shot < min_shot:
            return -1, -1, -1   # do not satisfy the minimum requirement.

        n_shot = 1 if (task_type == '1shot'
                       ) else 2 if (task_type == '2shot'
                                    ) else 5 if (task_type == '5shot'
                                                 ) else max_shot if (use_max_shot
                                                                     ) else np.random.randint(min_shot, max_shot + 1)

        available_cluster_idxs = check_available(num_imgs_clusters, n_way, n_shot, n_query)

        if len(available_cluster_idxs) < min_available_clusters:
            print(f"available_setting error with information: \n"
                  f"way [{min_way}, {max_way}]:{n_way} shot [{min_shot}, {max_shot}]:{n_shot}, \n"
                  f"pool: {num_imgs_clusters}, \n"
                  f"avail: {available_cluster_idxs}")
        else:
            break

    return n_way, n_shot, n_query


def check_available(num_imgs_clusters, n_way, n_shot, n_query):
    """Check whether pool has enough samples for specific setting and return available cluster idxes.
    :param num_imgs_clusters: list of Numpy array with shape [num_clusters * [num_classes]]
                              indicating number of images for specific class in specific clusters.
    :param n_way:
    :param n_shot:
    :param n_query:

    :return available cluster idxes which can satisfy sampling n_way, n_shot, n_query.
    """
    available_cluster_idxs = []
    for idx, num_imgs in enumerate(num_imgs_clusters):
        if len(num_imgs[num_imgs >= n_shot + n_query]) >= n_way:
            available_cluster_idxs.append(idx)

    return available_cluster_idxs


def task_to_device(task, d='numpy'):
    new_task = {}
    if d == 'numpy':
        new_task['context_images'] = task['context_images'].cpu().numpy()
        new_task['context_labels'] = task['context_labels'].cpu().numpy()
        new_task['target_images'] = task['target_images'].cpu().numpy()
        new_task['target_labels'] = task['target_labels'].cpu().numpy()
    else:
        new_task['context_images'] = torch.from_numpy(task['context_images']).to(d)
        new_task['context_labels'] = torch.from_numpy(task['context_labels']).long().to(d)
        new_task['target_images'] = torch.from_numpy(task['target_images']).to(d)
        new_task['target_labels'] = torch.from_numpy(task['target_labels']).long().to(d)

    return new_task


if __name__ == '__main__':

    pool = Pool(30000, 0)
    images = np.arange(100000).reshape(10000, 10)
    labels = np.repeat(np.arange(100), 100)
    tasks = np.repeat(np.arange(10), 1000)
    pool.put(images, {'labels': labels, 'tasks': tasks})

    num_imgs_clusters = [np.array([cls[1] for cls in classes]) for classes in pool.current_classes()]
    cluster_idx = 1
    n_way, n_shot, n_query = available_setting(
        [num_imgs_clusters[cluster_idx]], 'standard')
    print(n_way, n_shot, n_query)
    fs_task = pool.episodic_sample(cluster_idx, n_way, n_shot, n_query, d='cuda')

    # cal_hv_loss(np.array([[1,0], [0,1]]), 2)
    #
    # _n_way, _n_shot, _n_query = available_setting([np.array([20,20,20,20,20]), np.array([])], task_type='standard')
