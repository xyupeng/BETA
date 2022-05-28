import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from sklearn.mixture import GaussianMixture
from torchnet.meter import AUCMeter

from utils.util import AverageMeter, accuracy, TrackMeter
from utils.util import set_seed

from utils.config import Config, ConfigDict, DictAction
from losses import build_loss
from builder import build_optimizer
from models.build import build_model
from utils.util import format_time, interleave, de_interleave
from builder import build_logger
from datasets import build_divm_loader
from losses.TransLoss import AdversarialLoss


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


def adjust_lr(optimizer, step, tot_steps, gamma=10, power=0.75):
    decay = (1 + gamma * step / tot_steps) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['init_lr'] * decay


def set_optimizer(model, cfg):
    base_params = [v for k, v in model.named_parameters() if 'fc' not in k]
    head_params = [v for k, v in model.named_parameters() if 'fc' in k]
    param_groups = [{'params': base_params, 'lr': cfg.lr * 0.1},
                    {'params': head_params, 'lr': cfg.lr}]
    optimizer = build_optimizer(cfg.optimizer, param_groups)
    for param_group in optimizer.param_groups:
        param_group['init_lr'] = param_group['lr']
    return optimizer


def set_model(cfg):
    model = build_model(cfg.tgt_model)
    model.fc = build_model(cfg.tgt_head)
    model = torch.nn.DataParallel(model).cuda()
    return model


def update_batch_stats(model, flag):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.update_batch_stats = flag


def test(test_loader, model, criterion, epoch, logger, writer, model2=None):
    """ test target """
    model.eval()
    if model2 is not None:
        model2.eval()

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
            if model2 is not None:
                logits2 = model2(images)
                logits = (logits + logits2) / 2
            loss = criterion(logits, labels)

            pred = F.softmax(logits, dim=1)
            all_pred.append(pred.detach())

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

    all_pred = torch.cat(all_pred)
    mean_ent = (-all_pred * torch.log(all_pred + 1e-5)).sum(dim=1).mean().item() / np.log(all_pred.size(0))

    # writer
    writer.add_scalar(f'Loss/divm_test', losses.avg, epoch)
    writer.add_scalar(f'Entropy/divm_test', mean_ent, epoch)
    writer.add_scalar(f'Acc/divm_test', top1.avg, epoch)

    # logger
    time2 = time.time()
    test_time = format_time(time2 - time1)
    logger.info(f'Test at epoch [{epoch}] - test_time: {test_time}, '
                f'test_loss: {losses.avg:.3f}, '
                f'test_entropy: {mean_ent:.3f}, '
                f'test_Acc@1: {top1.avg:.2f}')
    return top1.avg, mean_ent


def test_class_acc(test_loader, model, criterion, it, logger, writer, cfg, model2=None):
    """ test target """
    model.eval()
    if model2 is not None:
        model2.eval()

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
            if model2 is not None:
                logits2 = model2(images)
                logits = (logits + logits2) / 2
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


def pred_target(test_loader, model, epoch, logger, cfg, model2=None):
    """ get predictions for target samples """
    model.eval()
    if model2 is not None:
        model2.eval()

    all_psl = []
    all_labels = []
    all_pred = []

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            bsz = images.shape[0]

            # forward
            logits = model(images)
            if model2 is not None:
                output2 = model2(images)
                logits = (logits + output2) / 2

            psl = logits.max(dim=1).indices
            pred = F.softmax(logits, dim=1)

            if epoch == 0:
                src_idx = torch.sort(pred, dim=1, descending=True).indices
                for i in range(bsz):
                    pred[i, src_idx[i, cfg.topk:]] = \
                        (1.0 - pred[i, src_idx[i, :cfg.topk]].sum()) / (cfg.num_classes - cfg.topk)

            all_psl.append(psl)
            all_labels.append(labels)
            all_pred.append(pred.detach())
    all_psl = torch.cat(all_psl)
    all_labels = torch.cat(all_labels)
    all_pred = torch.cat(all_pred)
    psl_acc = (all_psl == all_labels).float().mean()

    # logger
    time2 = time.time()
    pred_time = format_time(time2 - time1)
    logger.info(f'Predict target at epoch [{epoch}]: psl_acc: {psl_acc:.2f}, time: {pred_time}')
    return all_psl, all_labels, all_pred


def warmup(warmup_loader, model, optimizer, epoch, logger, cfg):
    batch_time = AverageMeter()
    losses = AverageMeter()

    criterion = nn.CrossEntropyLoss().cuda()
    num_iters = len(warmup_loader)

    model.train()
    t1 = end = time.time()
    for batch_idx, (inputs, labels) in enumerate(warmup_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                        f'Batch time: {batch_time.avg:.2f}, '
                        f'lr: {lr:.6f}, '
                        f'loss: {losses.avg:.3f}')

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(f'Epoch [{epoch}] - train_time: {epoch_time}, '
                f'train_loss: {losses.avg:.3f}\n')


def dist_train(warmup_loader, model, optimizer, epoch, logger, cfg, pred_mem):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_kl = AverageMeter()
    losses_ent = AverageMeter()

    num_iters = len(warmup_loader)

    model.train()
    t1 = end = time.time()
    for batch_idx, (images, _, indices) in enumerate(warmup_loader):
        images = images.cuda()
        targets = pred_mem[indices, :]
        bsz = images.shape[0]

        # forward
        logits = model(images)
        pred_tgt = F.softmax(logits, dim=1)
        loss_kl = nn.KLDivLoss(reduction='batchmean')(pred_tgt.log(), targets)

        loss_entropy = (-pred_tgt * torch.log(pred_tgt + 1e-5)).sum(dim=1).mean()
        pred_mean = pred_tgt.mean(dim=0)
        loss_gentropy = torch.sum(-pred_mean * torch.log(pred_mean + 1e-5))
        loss_entropy -= loss_gentropy
        loss = loss_kl + loss_entropy

        # update metric
        losses.update(loss.item(), bsz)
        losses_kl.update(loss_kl.item(), bsz)
        losses_ent.update(loss_entropy.item(), bsz)

        # backward1
        optimizer.zero_grad()
        loss.backward()

        # backward2
        if cfg.lam_mix > 0:
            alpha = 0.3
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(bsz).cuda()
            mixed_images = lam * images + (1 - lam) * images[index, :]
            mixed_targets = (lam * pred_tgt + (1 - lam) * pred_tgt[index, :]).detach()

            update_batch_stats(model, False)
            mixed_logits = model(mixed_images)
            update_batch_stats(model, True)
            mixed_pred_tgt = F.softmax(mixed_logits, dim=1)
            loss_mix_kl = cfg.lam_mix * nn.KLDivLoss(reduction='batchmean')(mixed_pred_tgt.log(), mixed_targets)
            loss_mix_kl.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                f'Batch time: {batch_time.avg:.2f}, '
                f'lr: {lr:.6f}, '
                f'loss_kl: {losses_kl.avg:.3f}, '
                f'loss_ent: {losses_ent.avg:.3f}, '
                f'distill_loss: {losses.avg:.3f}'
            )

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(
        f'Epoch [{epoch}] - train_time: {epoch_time}, '
        f'loss_kl: {losses_kl.avg:.3f}, '
        f'loss_ent: {losses_ent.avg:.3f}, '
        f'distill_loss: {losses.avg:.3f}'
    )


def eval_train(eval_loader, model):
    model.eval()
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):  # shuffle=False
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss)

    losses = torch.cat(losses)
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    losses = losses.cpu()

    # fit a two-component GMM to the loss
    input_loss = losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, losses


def train(label_loader, unlabel_loader, model, model2, criterion, optimizer, epoch, logger, cfg):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_trans = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    t1 = end = time.time()

    labeled_train_iter = iter(label_loader)
    unlabeled_train_iter = iter(unlabel_loader)

    adv = AdversarialLoss()

    model.train()
    model2.eval()
    num_iters = len(label_loader)
    for batch_idx in range(num_iters):
        try:
            (inputs_x1, inputs_x2), targets_x, w_x = next(labeled_train_iter)
        except:
            assert False
        try:
            (inputs_u1, inputs_u2) = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabel_loader)
            (inputs_u1, inputs_u2) = next(unlabeled_train_iter)
        batch_size = inputs_x1.size(0)

        # to cuda
        inputs_x1, inputs_x2 = inputs_x1.cuda(), inputs_x2.cuda()
        inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()
        targets_x = torch.zeros(batch_size, cfg.num_classes).scatter_(1, targets_x.view(-1, 1), 1).cuda()
        w_x = w_x.view(-1, 1).cuda()

        # co-refinement and co-guessing
        with torch.no_grad():
            # label refinement of labeled samples
            outputs_x1 = model(inputs_x1)
            outputs_x2 = model(inputs_x2)

            px = (torch.softmax(outputs_x1, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * targets_x + (1 - w_x) * px
            ptx = px ** (1 / cfg.T_sharpen)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

            # label co-guessing of unlabeled samples
            outputs_u11 = model(inputs_u1)
            outputs_u12 = model(inputs_u2)
            outputs_u21 = model2(inputs_u1)
            outputs_u22 = model2(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
                  torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu ** (1 / cfg.T_sharpen)

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

        # mixmatch forward
        lam = np.random.beta(cfg.alpha, cfg.alpha)
        lam = max(lam, 1 - lam)

        all_inputs = torch.cat([inputs_x1, inputs_x2, inputs_u1, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        lam_u = cfg.lam_u
        if cfg.lam_u > 0:
            mixed_input = lam * input_a + (1 - lam) * input_b
            mixed_target = lam * target_a + (1 - lam) * target_b

            mixed_inputs = interleave(mixed_input, batch_size)
            logits = model(mixed_inputs)
            logits = de_interleave(logits, batch_size)

            # loss
            Lx, Lu = criterion(
                logits[:batch_size * 2], mixed_target[:batch_size * 2],
                logits[batch_size * 2:], mixed_target[batch_size * 2:]
            )
            cur_epoch = epoch - 1 + batch_idx / num_iters
            lam_u = cfg.lam_u * np.clip((cur_epoch - cfg.warmup_epochs) / cfg.rampup_epochs, 0., 1.)
            loss = Lx + lam_u * Lu
            losses_u.update(Lu.item())
        else:
            mixed_input = lam * input_a[:batch_size * 2] + (1 - lam) * input_b[:batch_size * 2]
            mixed_target = lam * target_a[:batch_size * 2] + (1 - lam) * target_b[:batch_size * 2]

            mixed_inputs = interleave(mixed_input, batch_size)
            logits = model(mixed_inputs)
            logits = de_interleave(logits, batch_size)

            Lx = criterion(logits, mixed_target)  # SmoothCE
            loss = Lx
        losses_x.update(Lx.item())

        # penalty
        if cfg.lam_p > 0:
            prior = torch.ones(cfg.num_classes).cuda() / cfg.num_classes
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))
            loss += cfg.lam_p * penalty

        # transfer
        if cfg.lam_t > 0:
            _, feat_a = model(input_a, req_feat=True)
            _, feat_b = model(input_b, req_feat=True)
            transfer_loss = adv(feat_a, feat_b)
            loss += cfg.lam_t * transfer_loss
            losses_trans.update(transfer_loss.item())

        # update losses
        losses.update(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger
        if batch_idx % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch}][{batch_idx}/{num_iters}] - '
                        f'Batch time: {batch_time.avg:.2f}, '
                        f'lr: {lr:.6f}, '
                        f'loss: {losses.avg:.3f}, '
                        f'loss_trans: {losses_trans.avg:.3f}, '
                        f'loss_x: {losses_x.avg:.3f}, '
                        f'loss_u: {losses_u.avg:.3f}(lam_u={lam_u:.2f})')

    t2 = time.time()
    epoch_time = format_time(t2 - t1)
    logger.info(
        f'Epoch [{epoch}] - train_time: {epoch_time}, '
        f'loss_trans: {losses_trans.avg:.3f}, '
        f'loss_x: {losses_x.avg:.3f}, '
        f'loss: {losses.avg:.3f}'
    )
    return losses.avg, losses_x.avg, losses_u.avg


def main():
    # args & cfg
    args = parse_args()
    cfg = get_cfg(args)
    cudnn.benchmark = True

    # write cfg
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.cfg')
    with open(log_file, 'a') as f:
        f.write(cfg.pretty_text)

    # logger
    logger = build_logger(cfg.work_dir, cfgname=f'train')
    writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, f'tensorboard'))

    '''
    # -----------------------------------------
    # build model & optimizer
    # -----------------------------------------
    '''
    # build source model & load weights
    src_model = build_model(cfg.src_model)
    src_model = torch.nn.DataParallel(src_model).cuda()

    print(f'==> Loading checkpoint "{cfg.load}"')
    ckpt = torch.load(cfg.load, map_location='cuda')
    src_model.load_state_dict(ckpt['model_state'])

    # build target model
    model1 = set_model(cfg)
    model2 = set_model(cfg)

    optimizer1 = set_optimizer(model1, cfg)
    optimizer2 = set_optimizer(model2, cfg)

    train_criterion = build_loss(cfg.loss.train).cuda()
    test_criterion = build_loss(cfg.loss.test).cuda()
    print('==> Model built.')

    '''
    # -----------------------------------------
    # build dataset/dataloader
    # -----------------------------------------
    '''
    test_loader = build_divm_loader(cfg, mode='test')

    '''
    # -----------------------------------------
    # Test source model before distill 
    # -----------------------------------------
    '''
    if cfg.get('test_class_acc', False):
        test_class_acc(test_loader, src_model, test_criterion, 0, logger, writer, cfg)
    else:
        test(test_loader, src_model, test_criterion, 0, logger, writer)

    '''
    # -----------------------------------------
    # Predict target 
    # -----------------------------------------
    '''
    tgt_psl, gt_labels, pred_mem = pred_target(test_loader, src_model, 0, logger, cfg)
    warmup_loader = build_divm_loader(cfg, mode='warmup', psl=tgt_psl)
    warmup_loader_idx = build_divm_loader(cfg, mode='warmup', return_idx=True)
    eval_train_loader = build_divm_loader(cfg, mode='eval_train', psl=tgt_psl)

    '''
    # -----------------------------------------
    # Start target training
    # -----------------------------------------
    '''
    print("==> Start training...")
    model1.train()
    model2.train()

    test_meter = TrackMeter()
    start_epoch = 1

    for epoch in range(start_epoch, cfg.epochs + 1):
        adjust_lr(optimizer1, epoch, cfg.epochs, power=1.5)
        adjust_lr(optimizer2, epoch, cfg.epochs, power=1.5)

        # momentum update pred_mem
        if epoch % cfg.pred_interval == 0:
            _, _, pred_t = pred_target(test_loader, model1, epoch, logger, cfg, model2)
            pred_mem = cfg.ema * pred_mem + (1 - cfg.ema) * pred_t
            model1.train()
            model2.train()

        if epoch <= cfg.warmup_epochs:
            warmup(warmup_loader, model1, optimizer1, epoch, logger, cfg)
            warmup(warmup_loader, model2, optimizer2, epoch, logger, cfg)

        else:
            # distill loss
            logger.info(f'Start distill training at epoch [{epoch}]...')
            dist_train(warmup_loader_idx, model1, optimizer1, epoch, logger, cfg, pred_mem)
            dist_train(warmup_loader_idx, model2, optimizer2, epoch, logger, cfg, pred_mem)

            # calc GMM probs
            logger.info(f'==> Start evaluation at epoch [{epoch}]...')
            t1 = time.time()

            prob1, losses1 = eval_train(eval_train_loader, model1)
            prob2, losses2 = eval_train(eval_train_loader, model2)
            mask1 = prob1 >= cfg.tau_p
            mask2 = prob2 >= cfg.tau_p

            t2 = time.time()
            eval_time = format_time(t2 - t1)
            logger.info(f'==> Evaluation finished ({eval_time}).')

            # DivideMix
            label_indices = mask1.nonzero()[0]
            unlabel_indices = (~mask1).nonzero()[0]
            masked_probs = prob2[mask1]
            masked_psl = tgt_psl[mask1]
            label_loader = build_divm_loader(cfg, mode='label', indices=label_indices, probs=masked_probs, psl=masked_psl)
            unlabel_loader = build_divm_loader(cfg, mode='unlabel', indices=unlabel_indices)
            if len(label_loader) > 0 and len(unlabel_loader) > 0:
                train(label_loader, unlabel_loader, model1, model2, train_criterion, optimizer1,
                      epoch, logger, cfg)
            else:
                logger.info(f'Skip DivM for model_1 at epoch [{epoch}] - num_label: {len(label_indices)}, '
                            f'num_unlabel: {len(unlabel_indices)}.')

            label_indices = mask2.nonzero()[0]
            unlabel_indices = (~mask2).nonzero()[0]
            masked_probs = prob1[mask2]
            masked_psl = tgt_psl[mask2]
            label_loader = build_divm_loader(cfg, mode='label', indices=label_indices, probs=masked_probs, psl=masked_psl)
            unlabel_loader = build_divm_loader(cfg, mode='unlabel', indices=unlabel_indices)
            if len(label_loader) > 0 and len(unlabel_loader) > 0:
                train(label_loader, unlabel_loader, model2, model1, train_criterion, optimizer2,
                      epoch, logger, cfg)
            else:
                logger.info(f'Skip DivM for model_2 at epoch [{epoch}] - num_label: {len(label_indices)}, '
                            f'num_unlabel: {len(unlabel_indices)}.')

        if epoch % cfg.test_interval == 0 or epoch == cfg.epochs:
            if cfg.get('test_class_acc', False):
                test_acc, mean_ent, pred_max = \
                    test_class_acc(test_loader, model1, test_criterion, epoch, logger, writer, cfg, model2)
            else:
                test_acc, mean_ent = test(test_loader, model1, test_criterion, epoch, logger, writer, model2)
            test_meter.update(test_acc, idx=epoch)

    # We print the best test_acc but use the last checkpoint for fine-tuning.
    logger.info(f'Best test_Acc@1: {test_meter.max_val:.2f} (epoch={test_meter.max_idx}).')

    # save last
    model_path = os.path.join(cfg.work_dir, 'last.pth')
    state_dict = {
        'model1_state': model1.state_dict(),
        'model2_state': model2.state_dict(),
        'optimizer1_state': optimizer1.state_dict(),
        'optimizer2_state': optimizer2.state_dict(),
        'epochs': cfg.epochs
    }
    torch.save(state_dict, model_path)


if __name__ == '__main__':
    main()
