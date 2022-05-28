import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .transforms import office_home_train, office_home_test


def _tolist(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, (np.ndarray, torch.Tensor)):
        return x.tolist()
    else:
        raise TypeError


class OfficeHome(Dataset):
    def __init__(self, lines=None, info_path=None, transform=None, return_idx=False):
        if lines is None:
            assert info_path is not None
            with open(info_path) as f:
                lines = f.readlines()
        samples = []
        for line in lines:
            tokens = line.split()
            samples.append((tokens[0], int(tokens[1])))
        assert len(samples) > 0

        self.data = samples
        self.transform = transform
        self.return_idx = return_idx

    def __getitem__(self, index):
        path, target = self.data[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.return_idx:
            return img, target, index
        else:
            return img, target

    def __len__(self):
        return len(self.data)


class SubOfficeHome(Dataset):
    def __init__(self, info_path, transform=None, mode='train',
                 indices=None, probs=None, psl=None, return_idx=False):
        assert mode in ['warmup', 'eval_train', 'label', 'unlabel', 'test']

        sample_list = open(info_path).readlines()
        self.paths = []
        self.targets = []
        if indices is not None:
            sample_list = [sample_list[i] for i in indices]
        for i, line in enumerate(sample_list):
            tokens = line.split()
            self.paths.append(tokens[0])
            self.targets.append(int(tokens[1]))
        if psl is not None:
            self.targets = _tolist(psl)
        assert len(self.paths) > 0 and len(self.paths) == len(self.targets)

        self.transform = transform
        self.mode = mode
        self.indices = indices
        self.probs = probs  # GMM probs;
        self.return_idx = return_idx

    def __getitem__(self, index):
        if self.mode == 'warmup' or self.mode == 'eval_train' or self.mode == 'test':
            path, target = self.paths[index], self.targets[index]
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            if self.return_idx:
                return img, target, index
            else:
                return img, target
        elif self.mode == 'label':
            path, target, prob = self.paths[index], self.targets[index], self.probs[index]
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            return img, target, prob
        elif self.mode == 'unlabel':
            path = self.paths[index]
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            return img
        else:
            raise ValueError

    def __len__(self):
        return len(self.paths)


def build_office_home_loaders(cfg, loader_list=['src_train', 'src_val']):
    loader_dict = {}
    src_list = open(cfg.info_path).readlines()

    # set aside 3 samples per class for validation
    count = np.zeros(cfg.num_classes)
    src_train, src_val = [], []
    for line in src_list:
        tokens = line.strip().split(' ')
        c = int(tokens[1])
        if count[c] < 3:
            count[c] += 1
            src_val.append(line)
        else:
            src_train.append(line)

    if 'src_train' in loader_list:
        src_train_set = OfficeHome(src_train, transform=office_home_train(cfg.mean, cfg.std))
        loader_dict['src_train'] = DataLoader(src_train_set, batch_size=cfg.batch_size,
                                              shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    if 'src_val' in loader_list:
        src_val_set = OfficeHome(src_val, transform=office_home_test(cfg.mean, cfg.std))
        loader_dict['src_val'] = DataLoader(src_val_set, batch_size=cfg.batch_size,
                                            shuffle=False, num_workers=cfg.num_workers, drop_last=False)
    return loader_dict
