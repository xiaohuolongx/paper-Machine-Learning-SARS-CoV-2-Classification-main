import os
import sys
sys.path.insert(0, os.getcwd())
import copy
import argparse
import shutil
import time
import numpy as np
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel

from utils.history import History
from utils.dataloader import Mydataset, collate
from utils.train_utils import (
    train, validation, print_info, file2dict,
    init_random_seed, set_random_seed, resume_model
)
from utils.inference import init_model
from core.optimizers import *
from models.build import BuildNet


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--split-validation',
        action='store_true',
        help='whether to split validation set from training set. '
             'If set, K-fold cross-validation will be performed '
             '(K = 1/ratio, ratio is specified by --ratio)')
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.2,
        help='the proportion of the validation set to the training set '
             '(determines K in K-fold, e.g. 0.2 -> 5-fold)')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    # 读取配置文件（所有参数在后续多折中可能会被修改，所以用深拷贝）
    model_cfg, train_pipeline, _, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    print_info(model_cfg)

    # 基础保存目录
    backbone_type = model_cfg.get('backbone').get('type')
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    base_save_dir = os.path.join('logs', backbone_type, dirname)

    # 设置随机种子（整个流程只设置一次，保证数据 shuffle 可复现）
    seed = init_random_seed(args.seed)
    set_random_seed(seed, deterministic=args.deterministic)

    # 读取全部数据并 shuffle（若需要 K 折）
    total_annotations = "datas/train.txt"
    with open(total_annotations, encoding='utf-8') as f:
        total_datas = f.readlines()

    # 设备选择
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 非 K 折模式（原逻辑） ----
    if not args.split_validation:
        train_datas = total_datas.copy()
        test_annotations = 'datas/test.txt'
        with open(test_annotations, encoding='utf-8') as f:
            val_datas = f.readlines()

        # 单次训练
        single_train(args, model_cfg, train_pipeline,
                     data_cfg, lr_config, optimizer_cfg,
                     train_datas, val_datas, device, seed,
                     base_save_dir, fold_name=None)

    # ---- K 折交叉验证模式 ----
    else:
        # 确定折数
        K = int(1.0 / args.ratio)
        if K < 2:
            raise ValueError(f"ratio={args.ratio} gives K={K}, K must be >=2 for cross-validation.")

        # 全局 shuffle，保证每折划分确定且可复现
        rng = np.random.default_rng(seed)
        rng.shuffle(total_datas)

        total_nums = len(total_datas)
        fold_size = int(total_nums * args.ratio)   # 每折大小

        all_best_val_acc = []

        for fold in range(K):
            print(f"\n{'='*50}")
            print(f"  Starting Fold {fold+1}/{K}")
            print(f"{'='*50}")

            # 切分验证集和训练集
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold != K - 1 else total_nums
            val_datas = total_datas[val_start:val_end]
            train_datas = total_datas[:val_start] + total_datas[val_end:]

            # 深拷贝配置，防止字典被 pop 修改后影响下一折
            _model_cfg = copy.deepcopy(model_cfg)
            _train_pipeline = copy.deepcopy(train_pipeline)
            _data_cfg = copy.deepcopy(data_cfg)
            _lr_config = copy.deepcopy(lr_config)
            _optimizer_cfg = copy.deepcopy(optimizer_cfg)

            # 每折独立训练，目录名为 fold_0, fold_1, ...
            fold_name = f'fold_{fold}'
            single_train(args, _model_cfg, _train_pipeline,
                         _data_cfg, _lr_config, _optimizer_cfg,
                         train_datas, val_datas, device, seed,
                         base_save_dir, fold_name=fold_name,
                         all_best_val_acc=all_best_val_acc)

        # 汇总 K 折结果
        print("\n" + "="*50)
        print("K-Fold Cross-Validation Results")
        print("="*50)
        for i, acc in enumerate(all_best_val_acc):
            print(f"Fold {i+1}: best_val_acc = {acc:.4f}")
        mean_acc = np.mean(all_best_val_acc)
        std_acc = np.std(all_best_val_acc)
        print(f"\nMean ± Std: {mean_acc:.4f} ± {std_acc:.4f}")
        # 同时保存到文件
        summary_path = os.path.join(base_save_dir, 'cv_results.txt')
        with open(summary_path, 'w') as f:
            f.write(f"K = {K}\n")
            for i, acc in enumerate(all_best_val_acc):
                f.write(f"Fold {i+1}: {acc:.4f}\n")
            f.write(f"Mean: {mean_acc:.4f}\n")
            f.write(f"Std: {std_acc:.4f}\n")
        print(f"Summary saved to {summary_path}")


def single_train(args, model_cfg, train_pipeline,
                 data_cfg, lr_config, optimizer_cfg,
                 train_datas, val_datas, device, seed,
                 base_save_dir, fold_name=None,
                 all_best_val_acc=None):
    """
    执行一次完整的训练（用于单次或 K 折中的一折）
    fold_name: 若为 K 折，传入 'fold_0' 等；否则为 None
    all_best_val_acc: 列表，用于收集每折最佳精度
    """
    # 确定保存目录
    if fold_name is not None:
        save_dir = os.path.join(base_save_dir, fold_name)
    else:
        save_dir = base_save_dir

    meta = dict()
    meta['save_dir'] = save_dir
    meta['seed'] = seed

    # 初始化模型
    print('Initialize the weights.')
    model = BuildNet(model_cfg)
    if not data_cfg.get('train').get('pretrained_flag'):
        model.init_weights()
    if data_cfg.get('train').get('freeze_flag') and data_cfg.get('train').get('freeze_layers'):
        freeze_layers = ' '.join(list(data_cfg.get('train').get('freeze_layers')))
        print('Freeze layers : ' + freeze_layers)
        model.freeze_layers(data_cfg.get('train').get('freeze_layers'))

    if device != torch.device('cpu'):
        model = DataParallel(model, device_ids=[args.gpu_id])

    # 初始化优化器（注意 pop 会修改字典，所以外层已传入深拷贝）
    optimizer = eval('optim.' + optimizer_cfg.pop('type'))(params=model.parameters(), **optimizer_cfg)

    # 初始化学习率策略
    lr_update_func = eval(lr_config.pop('type'))(**lr_config)

    # 构建数据加载器（验证集使用训练pipeline的深拷贝，保证标签处理一致）
    train_dataset = Mydataset(train_datas, train_pipeline)
    val_dataset = Mydataset(val_datas, copy.deepcopy(train_pipeline))
    train_loader = DataLoader(
        train_dataset, shuffle=True,
        batch_size=data_cfg.get('batch_size'),
        num_workers=data_cfg.get('num_workers'),
        pin_memory=True, drop_last=True,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False,
        batch_size=data_cfg.get('batch_size'),
        num_workers=data_cfg.get('num_workers'),
        pin_memory=True, drop_last=True,
        collate_fn=collate
    )

    # 训练状态字典
    runner = dict(
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        iter=0,
        epoch=0,
        max_epochs=data_cfg.get('train').get('epoches'),
        max_iters=data_cfg.get('train').get('epoches') * len(train_loader),
        best_train_loss=float('INF'),
        best_val_acc=float(0),
        best_train_weight='',
        best_val_weight='',
        last_weight=''
    )
    meta['train_info'] = dict(
        train_loss=[],
        val_loss=[],
        train_acc=[],
        val_acc=[]
    )

    # 断点续训（仅在非 K 折时可用；K 折下忽略）
    if args.resume_from and fold_name is None:
        model, runner, meta = resume_model(model, runner, args.resume_from, meta)
    else:
        os.makedirs(save_dir, exist_ok=True)
        # 保存配置文件到该折目录（仅第一折或单次时保存，避免重复）
        if fold_name is None or (fold_name == 'fold_0' and not os.path.exists(os.path.join(save_dir, os.path.split(args.config)[1]))):
            shutil.copyfile(args.config, os.path.join(save_dir, os.path.split(args.config)[1]))
        model = init_model(model, data_cfg, device=device, mode='train')

    # 初始化历史记录器（每折独立绘制曲线）
    train_history = History(meta['save_dir'])

    # 学习率调度器初始化
    lr_update_func.before_run(runner)

    # 训练循环
    for epoch in range(runner.get('epoch'), runner.get('max_epochs')):
        lr_update_func.before_train_epoch(runner)
        train(model, runner, lr_update_func, device, epoch,
              data_cfg.get('train').get('epoches'), meta)
        validation(model, runner, data_cfg.get('test'), device, epoch,
                   data_cfg.get('train').get('epoches'), meta)
        train_history.after_epoch(meta)

    # 记录该折最佳验证精度
    if all_best_val_acc is not None:
        all_best_val_acc.append(runner['best_val_acc'])


if __name__ == "__main__":
    main()