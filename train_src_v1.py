import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from utils.util import AverageMeter, accuracy, TrackMeter
from utils.util import set_seed

from utils.config import Config, ConfigDict, DictAction
from losses import build_loss
from builder import build_optimizer
from models.build import build_model
from utils.util import format_time
from builder import build_logger
from datasets import build_dataset, build_office_home_loaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--load', type=str, help='Load init weights for fine-tune (default: None)')
    parser.add_argument('--cfgname', help='specify log_file; for debug use')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override the config; e.g., --cfg-options port=10001 k1=a,b k2="[a,b]"'
                             'Note that the quotation marks are necessary and that no white space is allowed.')
    args = parser.parse_args()
    return args


def get_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        dirname = os.path.dirname(args.config).replace('configs', 'checkpoints', 1)
        filename = os.path.splitext(os.path.basename(args.config))[0]
        cfg.work_dir = os.path.join(dirname, filename)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # cfgname
    if args.cfgname is not None:
        cfg.cfgname = args.cfgname
    else:
        cfg.cfgname = os.path.splitext(os.path.basename(args.config))[0]
    assert cfg.cfgname is not None

    # seed
    if args.seed != 0:
        cfg.seed = args.seed
    elif not hasattr(cfg, 'seed'):
        cfg.seed = 42
    set_seed(cfg.seed)

    # resume or load init weights
    if args.resume:
        cfg.resume = args.resume
    if args.load:
        cfg.load = args.load
    assert not (cfg.resume and cfg.load)

    return cfg


def adjust_lr(optimizer, it, train_iters, gamma=10, power=0.75):
    decay = (1 + gamma * it / train_iters) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['init_lr'] * decay


def val(val_loader, model, criterion, it, logger, writer):
    """validation"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

    # writer
    writer.add_scalar(f'Loss/src_val', losses.avg, it)
    writer.add_scalar(f'Acc/src_val', top1.avg, it)

    # logger
    time2 = time.time()
    val_time = format_time(time2 - time1)
    logger.info(f'Iter [{it}] - val_time: {val_time}, '
                f'val_loss: {losses.avg:.3f}, '
                f'val_Acc@1: {top1.avg:.2f}')
    return top1.avg


def main():
    # args & cfg
    args = parse_args()
    cfg = get_cfg(args)  # may modify cfg according to args
    cudnn.benchmark = True

    # write cfg
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.cfg')
    with open(log_file, 'a') as f:
        f.write(cfg.pretty_text)

    # logger
    logger = build_logger(cfg.work_dir, cfgname='train_source')
    writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, f'tensorboard'))

    '''
    # -----------------------------------------
    # build dataset/dataloader
    # -----------------------------------------
    '''
    loader_dict = build_office_home_loaders(cfg, loader_list=['src_train', 'src_val'])
    print(f'==> DataLoader built.')

    '''
    # -----------------------------------------
    # build model & optimizer
    # -----------------------------------------
    '''
    model = build_model(cfg.model)
    nn.init.xavier_normal_(model.fc.weight)
    model = torch.nn.DataParallel(model).cuda()
    train_criterion = build_loss(cfg.loss.train).cuda()
    val_criterion = build_loss(cfg.loss.val).cuda()

    base_params = [v for k, v in model.named_parameters() if 'fc' not in k]
    head_params = [v for k, v in model.named_parameters() if 'fc' in k]
    param_groups = [{'params': base_params, 'lr': cfg.lr * 0.1},
                    {'params': head_params, 'lr': cfg.lr}]
    optimizer = build_optimizer(cfg.optimizer, param_groups)
    for param_group in optimizer.param_groups:
        param_group['init_lr'] = param_group['lr']
    print('==> Model built.')

    '''
    # -----------------------------------------
    # Start source training
    # -----------------------------------------
    '''
    print("==> Start training...")
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    val_meter = TrackMeter()
    start_iter = 1
    train_iters = cfg.epochs * len(loader_dict['src_train'])
    val_interval = train_iters // 10

    end = time.time()
    iter_source = iter(loader_dict['src_train'])
    for it in range(start_iter, train_iters + 1):
        adjust_lr(optimizer, it, train_iters, power=0.75)

        try:
            images, labels = next(iter_source)
        except StopIteration:
            iter_source = iter(loader_dict['src_train'])
            images, labels = next(iter_source)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = images.shape[0]

        targets = torch.zeros(bsz, cfg.num_classes).cuda().scatter_(1, labels.view(-1, 1), 1)
        targets = (1 - cfg.eps) * targets + cfg.eps / cfg.num_classes

        # compute loss
        output = model(images)
        loss = train_criterion(output, targets)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))  # acc use labels
        top1.update(acc1[0], bsz)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if it == start_iter or it % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Iter [{it}/{train_iters}] - '
                        f'batch_time: {batch_time.avg:.3f},     '
                        f'lr: {lr:.5f},     '
                        f'loss: {losses.avg:.3f},     '
                        f'train_Acc@1: {top1.avg:.2f}')
            writer.add_scalar(f'lr', lr, it)
            writer.add_scalar(f'Loss/src_train', losses.avg, it)
            writer.add_scalar(f'Acc/src_train', top1.avg, it)

        if it % val_interval == 0 or it == train_iters:
            val_acc = val(loader_dict['src_val'], model, val_criterion, it, logger, writer)
            if val_acc >= val_meter.max_val:
                model_path = os.path.join(cfg.work_dir, f'best_val.pth')
                state_dict = {
                    'optimizer_state': optimizer.state_dict(),
                    'model_state': model.state_dict(),
                    'iter': it
                }
                torch.save(state_dict, model_path)

            val_meter.update(val_acc, idx=it)
            logger.info(f'Best val_Acc@1: {val_meter.max_val:.2f} (iter={val_meter.max_idx}).')
            model.train()

    # save last
    model_path = os.path.join(cfg.work_dir, 'last.pth')
    state_dict = {
        'optimizer_state': optimizer.state_dict(),
        'model_state': model.state_dict(),
        'iter': train_iters
    }
    torch.save(state_dict, model_path)


if __name__ == '__main__':
    main()
