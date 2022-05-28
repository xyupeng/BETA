import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.util import AverageMeter, accuracy, TrackMeter
from utils.util import set_seed

from utils.config import Config, ConfigDict, DictAction
from losses import build_loss
from builder import build_optimizer
from models.build import build_model
from utils.util import format_time
from builder import build_logger
from datasets import build_dataset


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

    # resume or load init weights
    if args.resume:
        cfg.resume = args.resume
    if args.load:
        cfg.load = args.load
    assert not (cfg.resume and cfg.load)

    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.dirname(cfg.load)
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


    return cfg


def adjust_lr(optimizer, it, train_iters, gamma=10, power=0.75):
    decay = (1 + gamma * it / train_iters) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['init_lr'] * decay


def update_batch_stats(model, flag):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.update_batch_stats = flag


def test(test_loader, model, criterion, it, logger, writer):
    """ test target """
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    all_pred = []

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            logits = model(images)
            loss = criterion(logits, labels)

            pred = F.softmax(logits, dim=1)
            all_pred.append(pred.detach())

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

    all_pred = torch.cat(all_pred)
    mean_ent = (-all_pred * torch.log(all_pred + 1e-5)).sum(dim=1).mean().item() / np.log(all_pred.size(0))
    pred_max = all_pred.max(dim=1).indices

    # writer
    writer.add_scalar(f'Loss/ft_tgt_test', losses.avg, it)
    writer.add_scalar(f'Entropy/ft_tgt_test', mean_ent, it)
    writer.add_scalar(f'Acc/ft_tgt_test', top1.avg, it)

    # logger
    time2 = time.time()
    test_time = format_time(time2 - time1)
    logger.info(f'Test at iter [{it}] - test_time: {test_time}, '
                f'test_loss: {losses.avg:.3f}, '
                f'test_entropy: {mean_ent:.3f}, '
                f'test_Acc@1: {top1.avg:.2f}')
    return top1.avg, mean_ent, pred_max


def test_class_acc(test_loader, model, criterion, it, logger, writer, cfg):
    """ test target """
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    all_pred, all_labels = [], []

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            all_labels.append(labels)
            bsz = labels.shape[0]

            # forward
            logits = model(images)
            loss = criterion(logits, labels)

            pred = F.softmax(logits, dim=1)
            all_pred.append(pred.detach())

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

    all_labels = torch.cat(all_labels)
    all_pred = torch.cat(all_pred)
    mean_ent = (-all_pred * torch.log(all_pred + 1e-5)).sum(dim=1).mean().item() / np.log(all_pred.size(0))
    pred_max = all_pred.max(dim=1).indices

    # class-wise acc
    class_accs = []
    all_eq = pred_max == all_labels
    for c in range(cfg.num_classes):
        mask_c = all_labels == c
        acc_c = all_eq[mask_c].float().mean().item()
        class_accs.append(round(acc_c * 100, 2))
    avg_acc = round(sum(class_accs) / len(class_accs), 2)

    # writer
    writer.add_scalar(f'Loss/ft_tgt_test', losses.avg, it)
    writer.add_scalar(f'Entropy/ft_tgt_test', mean_ent, it)
    writer.add_scalar(f'Acc/ft_tgt_test', top1.avg, it)

    # logger
    time2 = time.time()
    test_time = format_time(time2 - time1)
    logger.info(f'Test at iter [{it}] - test_time: {test_time}, '
                f'test_loss: {losses.avg:.3f}, '
                f'test_entropy: {mean_ent:.3f}, '
                f'test_Acc@1: {top1.avg:.2f}')
    logger.info(f'per class acc: {str(class_accs)}, avg_acc: {avg_acc}')
    return top1.avg, mean_ent, pred_max


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
    logger = build_logger(cfg.work_dir, cfgname=f'finetune')
    writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, f'tensorboard'))

    '''
    # -----------------------------------------
    # build dataset/dataloader
    # -----------------------------------------
    '''
    # loader_dict = build_office_home_loaders(cfg, tgt=cfg.tgt, loader_list=['tgt_train', 'tgt_test'])
    loader_dict = {}
    train_set = build_dataset(cfg.data.train)
    test_set = build_dataset(cfg.data.test)
    loader_dict['tgt_train'] = DataLoader(train_set, batch_size=cfg.batch_size,
                                          shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    loader_dict['tgt_test'] = DataLoader(test_set, batch_size=cfg.batch_size,
                                         shuffle=False, num_workers=cfg.num_workers, drop_last=False)
    print(f'==> DataLoader built.')

    '''
    # -----------------------------------------
    # build model & optimizer
    # -----------------------------------------
    '''
    # build target model & load weights
    model = build_model(cfg.tgt_model)
    model.fc = build_model(cfg.tgt_head)
    model = torch.nn.DataParallel(model).cuda()

    print(f'==> Loading checkpoint "{cfg.load}"')
    ckpt = torch.load(cfg.load, map_location='cuda')
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt.keys() else ckpt['model1_state'])
    test_criterion = build_loss(cfg.loss.test).cuda()

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
    # Test distilled model before finetune 
    # -----------------------------------------
    '''
    if cfg.get('test_class_acc', False):
        test_class_acc(loader_dict['tgt_test'], model, test_criterion, 0, logger, writer, cfg)
    else:
        test(loader_dict['tgt_test'], model, test_criterion, 0, logger, writer)

    '''
    # -----------------------------------------
    # Start target training (finetune)
    # -----------------------------------------
    '''
    print("==> Start training...")
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    test_meter = TrackMeter()
    start_iter = 1
    train_iters = cfg.epochs * len(loader_dict['tgt_train'])
    test_interval = train_iters // 10
    last_pred = -1

    end = time.time()
    iter_source = iter(loader_dict['tgt_train'])
    for it in range(start_iter, train_iters + 1):
        # train
        adjust_lr(optimizer, it, train_iters, power=0.75)

        try:
            images, labels = next(iter_source)
        except StopIteration:
            iter_source = iter(loader_dict['tgt_train'])
            images, labels = next(iter_source)

        images = images.cuda(non_blocking=True)
        bsz = images.shape[0]

        # forward
        logits = model(images)
        pred_tgt = F.softmax(logits, dim=1)

        loss_entropy = (-pred_tgt * torch.log(pred_tgt + 1e-5)).sum(dim=1).mean()
        pred_mean = pred_tgt.mean(dim=0)
        loss_gentropy = torch.sum(-pred_mean * torch.log(pred_mean + 1e-5))
        loss_entropy -= loss_gentropy
        loss = loss_entropy

        # update metric
        losses.update(loss.item(), bsz)

        # backward1
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
                        f'loss: {losses.avg:.3f}')
            writer.add_scalar(f'lr/ft_tgt', lr, it)
            writer.add_scalar(f'Loss/ft_tgt_train', losses.avg, it)

        if it % test_interval == 0 or it == train_iters:
            if cfg.get('test_class_acc', False):
                test_acc, mean_ent, pred_max = \
                    test_class_acc(loader_dict['tgt_test'], model, test_criterion, it, logger, writer, cfg)
            else:
                test_acc, mean_ent, pred_max = \
                    test(loader_dict['tgt_test'], model, test_criterion, it, logger, writer)
            test_meter.update(test_acc, idx=it)
            model.train()

            if torch.abs(pred_max - last_pred).sum() == 0:
                break
            last_pred = pred_max

    # We print the best test_acc but report test_acc of the last epoch.
    logger.info(f'Best test_Acc@1: {test_meter.max_val:.2f} (iter={test_meter.max_idx}).')

    # save last
    model_path = os.path.join(cfg.work_dir, 'ft_last.pth')
    state_dict = {
        'optimizer_state': optimizer.state_dict(),
        'model_state': model.state_dict(),
        'iter': train_iters
    }
    torch.save(state_dict, model_path)


if __name__ == '__main__':
    main()
