import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features  # prototype
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())  # Similarity between batch feature and prototype

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:  # True
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


def anchor(batch_input, batch_labels, indexes, feature_memory, k, temp, momentum):
    """
    The anchor contrastive loss implementation in our paper.(Andy Zhu)
    The idea is to find the hardest same cluster sample as positive sample. (the minimum cosine similarity)
    And find K hard different cluster samples as negative samples. (the maximum cosine similarity)
    Finally, update feature memory by momentum update.
    """
    instance_m = feature_memory.features.clone().detach()
    mat = torch.matmul(batch_input, instance_m.transpose(0, 1))
    positives = []
    negatives = []
    for i in range(batch_labels.size(0)):
        pos_labels = (feature_memory.labels == batch_labels[i])
        pos = mat[i, pos_labels]
        positives.append(pos[torch.argmin(pos)])
        neg_labels = (feature_memory.labels != batch_labels[i])  # pseudo labels w/o ignore
        # neg_labels = torch.logical_and(feature_memory.labels != batch_labels[i], feature_memory.labels != -1)  # ignore -1
        neg = torch.sort(mat[i, neg_labels], descending=True)[0]
        idx = neg[:k]
        negatives.append(idx)
    positives = torch.stack(positives)
    positives = positives.view(-1,1)
    negatives = torch.stack(negatives)
    anchor_out = torch.cat((positives, negatives), dim=1) / temp

    with torch.no_grad():
        for data, index in zip(batch_input, indexes):
            feature_memory.features[index] = momentum * feature_memory.features[index] + (1.-momentum) * data
            feature_memory.features[index] /= feature_memory.features[index].norm()
    return anchor_out


def path_refine(inputs_tuple, patch_rate, temp):
    cls = inputs_tuple[1].unsqueeze(1)
    tokens = inputs_tuple[2]
    mat = torch.einsum("bxd,byd->bxy", [cls, tokens])
    positives = []
    negatives = []
    B = cls.size(0)
    rate = int(tokens.size(1) * patch_rate)
    for i in range(B):
        index = torch.argmax(mat[i,:])
        positives.append(mat[i,:,index])
        neg = torch.sort(mat[i,:])[0]
        index_n = neg[:,:rate]
        negatives.append(index_n)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives).squeeze(1)
    patch_out = torch.cat((positives, negatives), dim=1) / temp
    return patch_out


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.criterion = nn.CrossEntropyLoss()
        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs_tuple, targets, indexes, feature_memory, k, patch_rate):
        inputs = inputs_tuple[0]
        inputs = F.normalize(inputs, dim=1).cuda()  # batch data
        contrast_targets = torch.zeros([targets.size(0)]).cuda().long()
        patch_out = path_refine(inputs_tuple, patch_rate, self.temp)
        patch_loss = self.criterion(patch_out, contrast_targets)
        anchor_out = anchor(inputs, targets, indexes, feature_memory, k, self.temp, self.momentum)
        anchor_loss = self.criterion(anchor_out, contrast_targets)

        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        # return loss
        return loss+anchor_loss+patch_loss
