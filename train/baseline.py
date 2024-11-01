# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta
import os
from sklearn.cluster import DBSCAN
import pandas as pd
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from my import datasets, models
from my.models.cm import ClusterMemory
from my.trainers_baseline import Trainer
from my.evaluators import Evaluator, extract_features
from my.utils.data import IterLoader
from my.utils.data import transforms as T
from my.utils.data.sampler import RandomMultipleGallerySampler
from my.utils.data.preprocessor import Preprocessor
from my.utils.logging import Logger
from my.utils.serialization import load_checkpoint, save_checkpoint
from my.utils.faiss_rerank import compute_jaccard_distance
start_epoch = best_mAP = 0


def get_data(name, data_dir):
    dataset = datasets.create(name, data_dir)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):
    if args.self_norm:
        normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    else:
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, nv_root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)  # shuffle=not rmgs_flag
    return train_loader


def get_test_loader(args, dataset, height, width, batch_size, workers, testset=None):
    if args.self_norm:
        normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    else:
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    if 'resnet' in args.arch:
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                num_classes=0, pooling_type=args.pooling_type,pretrained_path=args.pretrained_path)
    else:
        model = models.create(args.arch,img_size=(args.height,args.width),drop_path_rate=args.drop_path_rate
                , pretrained_path = args.pretrained_path,hw_ratio=args.hw_ratio, conv_stem=args.conv_stem, feat_fusion=args.feat_fusion, multi_neck=args.multi_neck)
    # use CUDA

    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(args, dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    print('optimizer: %s'%(args.optimizer))
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = Trainer(model)
    print('==> Initialize features memory')
    feature_memory = ClusterMemory(model.module.num_features, len(dataset.train), temp=args.temp,
                                   momentum=args.momentum, use_hard=args.use_hard).cuda()
    cluster_loader = get_test_loader(args, dataset, args.height, args.width,
                                     args.batch_size, args.workers, testset=sorted(dataset.train))

    # create index corresponding dic
    index_dic = {}
    for it, data in enumerate(cluster_loader):
        _, path_list, _, _, indexes = data
        for m in range(len(path_list)):
            file_path = path_list[m]
            file_name = file_path.split('/')[-1]
            index_dic[file_name] = indexes[m]

    features, _ = extract_features(model, cluster_loader, print_freq=50)
    features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
    feature_memory.features = F.normalize(features, dim=1).cuda()
    trainer.feature_memory = feature_memory
    # DBSCAN cluster init
    eps = args.eps
    print('Clustering criterion: eps: {:.3f}'.format(eps))
    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
    for epoch in range(args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            features, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            feature_memory.labels = torch.Tensor(pseudo_labels).cuda()
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features)
        del features

        # Create ClusterMemory
        memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()

        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset)

        train_loader.new_epoch()

        trainer.train(epoch, train_loader, optimizer, args.K, args.temp, args.momentum, index_dic=index_dic,
                      print_freq=args.print_freq, train_iters=len(train_loader), )

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='msmt17',  # msmt17, msmt17_v2, market1501
                        choices=datasets.names())
    parser.add_argument('--gpu', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-K', type=int, default=8192, help="negative samples number for instance memory")
    parser.add_argument('--patch-rate', type=float, default=0.025, help="noise patch rate for patch refine")
    parser.add_argument('--positive-rate', type=int, default=3, help="positive sample number for patch refine")
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each i dentity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # DBSCAN
    parser.add_argument('--eps', type=float, default=0.7,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='vit_small',
                        choices=models.names())
    parser.add_argument('-pp', '--pretrained-path', type=str, default='/home/andy/ICASSP_data/pretrain/PASS/pass_vit_small_full.pth')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the memory")
    #vit
    parser.add_argument('--drop-path-rate', type=float, default=0.3)
    parser.add_argument('--hw-ratio', type=int, default=2)
    parser.add_argument('--self-norm', default=True)
    parser.add_argument('--conv-stem', action="store_true")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/andy/ICASSP_data/data/')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default='./log/cluster_contrast_reid/msmt17_v1/baseline')  # msmt17_v1, market1501
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--feat-fusion', type=str, default='cat')
    # parser.add_argument('--multi-neck', default=True)
    parser.add_argument('--multi-neck', action="store_true")
    parser.add_argument('--use-hard', default=True)
    main()
