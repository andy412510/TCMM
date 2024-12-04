# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
from datetime import timedelta
import os
from collections import OrderedDict, Counter
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from TCMM.utils import to_torch
from TCMM import datasets, models
from TCMM.evaluators import Evaluator
from TCMM.utils.data import IterLoader
from TCMM.utils.data import transforms as T
from TCMM.utils.data.sampler import RandomMultipleGallerySampler
from TCMM.utils.data.preprocessor import Preprocessor
from TCMM.utils.logging import Logger
from TCMM.utils.serialization import load_checkpoint, save_checkpoint
from TCMM.utils.faiss_rerank import compute_jaccard_distance

from openTSNE import TSNE
import matplotlib.pyplot as plt
from examples import utils
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


def extract_features(model, data_loader, print_freq=50, cluster_features=True):
    # value: 1201, count number: 193
    # value: 2088, count number: 188
    # value: 702, count number: 168
    # value: 557, count number: 162
    # value: 1153, count number: 151
    # value: 1885, count number: 150
    # value: 411, count number: 148
    # value: 2710, count number: 134
    # value: 1186, count number: 134
    # value: 1461, count number: 134
    # value: 1896, count number: 132
    # value: 1895, count number: 132
    # value: 1660, count number: 132
    # value: 1606, count number: 130
    # value: 1561, count number: 130
    # value: 1615, count number: 129
    # value: 715, count number: 127
    # value: 3045, count number: 127
    # value: 2287, count number: 126
    # value: 2608, count number: 125
    model.eval()
    features = OrderedDict()
    labels = OrderedDict()
    select_features = OrderedDict()
    select_labels = OrderedDict()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            # test data feature extract
            imgs = to_torch(imgs).cuda()
            outputs = model(imgs)
            outputs = outputs[0].data.cpu()
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid  # file name = label
        for key, value in labels.items():
            if value.item() in [1201, 2088, 1153, 411, 2710, 1186]:  # 1201, 2088, 557, 1153, 411
                select_features[key] = features[key]
                # select_labels[key] = value
                if value.item() == 1201:
                    select_labels[key] = torch.tensor(0)
                elif value.item() == 2088:
                    select_labels[key] = torch.tensor(1)
                elif value.item() == 557:
                    select_labels[key] = torch.tensor(2)
                elif value.item() == 1153:
                    select_labels[key] = torch.tensor(3)
                elif value.item() == 411:
                    select_labels[key] = torch.tensor(4)
                elif value.item() == 2710:
                    select_labels[key] = torch.tensor(5)
                elif value.item() == 1186:
                    select_labels[key] = torch.tensor(6)
                elif value.item() == 1461:
                    select_labels[key] = torch.tensor(7)

    return select_features, select_labels

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
    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, '512_K4_r0.075_outlers.pth.tar'))  # 512_K4_r0.075_outlers
    model.load_state_dict(checkpoint['state_dict'])

    # get features
    select_features, select_labels = extract_features(model, test_loader, cluster_features=False)
    features_list = torch.stack([value for value in select_features.values()])
    labels_list = [value.item() for value in select_labels.values()]
    # DBSCAN cluster init
    eps = args.eps
    print('Clustering criterion: eps: {:.3f}'.format(eps))
    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
    rerank_dist = compute_jaccard_distance(features_list, k1=args.k1, k2=args.k2)


    # t-SNE
    X = np.array(features_list)
    # Y = np.array(labels_list)
    Y = cluster.fit_predict(rerank_dist)  # DBSCAN
    tsne = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )
    embedding_train = tsne.fit(X)  # tsne.fit(data feature)
    utils.plot(embedding_train, Y, colors=utils.MOUSE_10X_COLORS)  # (embedding, data label, color)
    plt.savefig('tsne.jpg')

    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='msmt17',  # msmt17, msmt17_v2, market1501
                        choices=datasets.names())
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default='/home/andy/main_code/train/log/cluster_contrast_reid/msmt17_v1')  # msmt17_v1, market1501
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('-b', '--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-K', type=int, default=8, help="negative samples number for instance memory")
    parser.add_argument('--patch-rate', type=float, default=0.025, help="noise patch rate for patch refine")
    parser.add_argument('--positive-rate', type=int, default=3, help="positive sample number for patch refine")
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each i dentity has num_instances instances, "
                             "default: 0 (NOT USE)")
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

    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--feat-fusion', type=str, default='cat')
    # parser.add_argument('--multi-neck', default=True)
    parser.add_argument('--multi-neck', action="store_true")
    parser.add_argument('--use-hard', default=True)
    # DBSCAN
    parser.add_argument('--eps', type=float, default=0.7,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    main()
