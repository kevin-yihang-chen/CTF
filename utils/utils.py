import collections
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader, RandomSampler, Sampler, SequentialSampler, WeightedRandomSampler,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def collate_MIL(batch):
    """Collate function for survival (8-tuple) and grading (5-tuple) datasets."""
    img = torch.cat([item[0] for item in batch], dim=0)
    img2 = torch.cat([item[1] for item in batch], dim=0)
    try:
        img3 = torch.cat([item[2] for item in batch], dim=0)
    except Exception:
        img3 = torch.zeros(1)

    if len(batch[0]) == 8:  # survival branch
        censor = torch.LongTensor([item[3] for item in batch])
        survival_months = torch.LongTensor([item[4] for item in batch])
        label = torch.LongTensor([item[5] for item in batch])
        slide_id = [item[6] for item in batch]
        coords = np.vstack([item[7] for item in batch])
        return [img, img2, img3, censor, survival_months, label, slide_id, coords]

    label = torch.LongTensor([item[2] for item in batch])
    slide_id = [item[3] for item in batch]
    coords = np.vstack([item[4] for item in batch])
    return [img, img2, label, slide_id, coords]


def get_split_loader(split_dataset, training=False, testing=False, weighted=False):
    kwargs = {'num_workers': 4} if device.type == 'cuda' else {}
    if testing:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset) * 0.1)), replace=False)
        return DataLoader(split_dataset, batch_size=1,
                          sampler=SubsetSequentialSampler(ids),
                          collate_fn=collate_MIL, **kwargs)
    if training:
        if weighted:
            weights = make_weights_for_balanced_classes_split(split_dataset)
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = RandomSampler(split_dataset)
    else:
        sampler = SequentialSampler(split_dataset)
    return DataLoader(split_dataset, batch_size=1, sampler=sampler,
                      collate_fn=collate_MIL, **kwargs)


def get_optim(model, args):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.opt == 'adam':
        return optim.Adam(params, lr=args.lr, weight_decay=args.reg)
    if args.opt == 'sgd':
        return optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.reg)
    raise NotImplementedError(args.opt)


def generate_split(cls_ids, val_num, test_num, samples,
                   n_splits=5, seed=7, custom_test_ids=None):
    indices = np.arange(samples).astype(int)
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for _ in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []
        if custom_test_ids is not None:
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible = np.intersect1d(cls_ids[c], indices)
            val_ids = np.random.choice(possible, val_num[c], replace=False)
            remaining = np.setdiff1d(possible, val_ids)
            all_val_ids.extend(val_ids)
            if custom_test_ids is None:
                test_ids = np.random.choice(remaining, test_num[c], replace=False)
                remaining = np.setdiff1d(remaining, test_ids)
                all_test_ids.extend(test_ids)
            sampled_train_ids.extend(remaining)

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y):
    return 1.0 - Y_hat.float().eq(Y.float()).float().mean().item()


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0.0] * int(N)
    for idx in range(len(dataset)):
        y = int(dataset.getlabel(idx))
        weight[idx] = weight_per_class[y]
    return torch.DoubleTensor(weight)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
