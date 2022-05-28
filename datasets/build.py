import copy
from torch.utils.data import DataLoader

import datasets
from datasets.transforms import build_transform
import torchvision


def build_dataset(cfg):
    # args = cfg.copy()  # if bug, may need deepcopy
    args = copy.deepcopy(cfg)

    # build transform
    transform = build_transform(args.trans_dict)

    # build dataset
    ds_dict = args.ds_dict
    ds_name = ds_dict.pop('type')
    ds_dict['transform'] = transform
    if hasattr(torchvision.datasets, ds_name):
        ds = getattr(torchvision.datasets, ds_name)(**ds_dict)
    else:
        ds = datasets.__dict__[ds_name](**ds_dict)
    return ds


def build_divm_loader(cfg, mode='train', indices=None, probs=None, psl=None, return_idx=False):
    if mode == 'warmup':
        warmup_cfg = copy.deepcopy(cfg.data.warmup)
        warmup_cfg.ds_dict.psl = psl
        warmup_cfg.ds_dict.return_idx = return_idx

        warmup_set = build_dataset(warmup_cfg)
        warmup_loader = DataLoader(
            warmup_set, batch_size=cfg.batch_size,
            shuffle=True, num_workers=cfg.num_workers, drop_last=True
        )
        return warmup_loader
    if mode == 'eval_train':
        eval_train_cfg = copy.deepcopy(cfg.data.eval_train)
        eval_train_cfg.ds_dict.psl = psl

        eval_train_set = build_dataset(eval_train_cfg)
        eval_train_loader = DataLoader(
            eval_train_set, batch_size=cfg.batch_size,
            shuffle=False, num_workers=cfg.num_workers, drop_last=False
        )
        return eval_train_loader
    elif mode == 'label':
        label_cfg = copy.deepcopy(cfg.data.label)
        label_cfg.ds_dict.indices = indices
        label_cfg.ds_dict.probs = probs
        label_cfg.ds_dict.psl = psl

        label_set = build_dataset(label_cfg)
        label_loader = DataLoader(
            label_set, batch_size=cfg.batch_size,
            shuffle=True, num_workers=cfg.num_workers, drop_last=True
        )
        return label_loader
    elif mode == 'unlabel':
        unlabel_cfg = copy.deepcopy(cfg.data.unlabel)
        unlabel_cfg.ds_dict.indices = indices

        unlabel_set = build_dataset(unlabel_cfg)
        unlabel_loader = DataLoader(
            unlabel_set, batch_size=cfg.batch_size,
            shuffle=True, num_workers=cfg.num_workers, drop_last=True
        )
        return unlabel_loader
    elif mode == 'test':
        test_set = build_dataset(cfg.data.test)
        test_loader = DataLoader(
            test_set, batch_size=cfg.batch_size,
            shuffle=False, num_workers=cfg.num_workers, drop_last=False
        )
        return test_loader
    else:
        raise ValueError

