from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import torch
import os
from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch
### vis_actmap ###
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def vis_attention(tensor, fnames, model):
    path = './vit_heatmap/'
    patch_size = 16
    for i in range(len(fnames)):
        fname = os.path.splitext(os.path.basename(fnames[i]))[0]
        input_tensor = tensor[i,:]
        w, h = input_tensor.shape[1] - input_tensor.shape[1] % patch_size, input_tensor.shape[2] - input_tensor.shape[2] % patch_size
        img = input_tensor[:, :w, :h].unsqueeze(0)
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        _, _, _, _, _, y = model.module.forward_features(img)
        attentions = model.module.get_last_selfattention(y)
        nh = attentions.shape[1]
        attentions = attentions[0, :, 0, 5:].reshape(nh, -1)
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[
            0].cpu().numpy()

        # save attention heatmap images
        torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True),
                                     os.path.join(path, fname + ".png"))
        fpath = os.path.join(path, fname + "_head" + '.png')
        # save last head, here nh=6, so save attentions[5]
        plt.imsave(fname=fpath, arr=attentions[5], format='png')


def extract_features(model, data_loader, query_loader, print_freq=50, cluster_features=True):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    img_paths = OrderedDict() #
    
    end = time.time()

    # ViT heatmap vis
    for p in model.parameters():
        p.requires_grad = False
    for i, (imgs, fnames, _, _, is_query) in enumerate(query_loader):  # query_loader only contains query data
        imgs = to_torch(imgs).cuda()
        vis_attention(imgs, fnames, model)
        print(f"batch {i} saved.")
    # ViT heatmap vis end


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m, x.numpy(), y.numpy(), list(features.keys())

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False, img_paths=None):
    query_img_paths = [img_paths[q[0]] for q in query] if query else []
    gallery_img_paths = [img_paths[g[0]] for g in gallery] if gallery else []

    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams, query_img_paths=query_img_paths, gallery_img_paths=gallery_img_paths)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query_loader, query, gallery, cmc_flag=False, rerank=False):
        extract_features(self.model, data_loader, query_loader, cluster_features=False)
