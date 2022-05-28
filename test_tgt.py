import os
import argparse
import time

import torch
import torch.backends.cudnn as cudnn

from utils.util import AverageMeter, accuracy, TrackMeter
from utils.util import set_seed

from utils.config import Config, ConfigDict, DictAction
from losses import build_loss
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
        cfg.work_dir = os.path.dirname(args.load)
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


def test(test_loader, model, criterion, it, logger):
    """ test target """
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    time1 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            logits = model(images)
            loss = criterion(logits, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

    # logger
    time2 = time.time()
    test_time = format_time(time2 - time1)
    logger.info(f'Iter [{it}] - test_time: {test_time}, '
                f'test_loss: {losses.avg:.3f}, '
                f'test_Acc@1: {top1.avg:.2f}')
    return top1.avg


def main():
    # args & cfg
    args = parse_args()
    cfg = get_cfg(args)  # may modify cfg according to args
    cudnn.benchmark = True

    # logger
    logger = build_logger(cfg.work_dir, cfgname='test_target')

    '''
    # -----------------------------------------
    # build model & optimizer
    # -----------------------------------------
    '''
    model = build_model(cfg.model)
    model = torch.nn.DataParallel(model).cuda()
    test_criterion = build_loss(cfg.loss.test).cuda()

    # load weights
    print(f'==> Loading checkpoint "{cfg.load}"')
    ckpt = torch.load(cfg.load, map_location='cuda')
    model.load_state_dict(ckpt['model_state'])
    test_iter = ckpt['iter']

    print('==> Model built.')

    '''
    # -----------------------------------------
    # Test
    # -----------------------------------------
    '''
    for dom in cfg.test_domains:
        logger.info(f'==> Testing domain "{dom}"...')
        loader_dict = build_office_home_loaders(cfg, tgt=dom, loader_list=['tgt_test'])
        test(loader_dict['tgt_test'], model, test_criterion, test_iter, logger)


if __name__ == '__main__':
    main()
