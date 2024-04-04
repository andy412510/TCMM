from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
from torchvision import transforms


class Trainer(object):
    def __init__(self, encoder, memory=None, feature_memory=None):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.feature_memory = feature_memory

    def train(self, epoch, data_loader, optimizer, k, patch_rate, positive_rate, index_dic, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes, path_list = self._parse_data(inputs)
            # correct index
            for j in range(len(path_list)):
                file_path = path_list[j]
                file_name = file_path.split('/')[-1]
                indexes[j] = index_dic[file_name]
            # forward
            f_out = self._forward(inputs)
            loss = self.memory(f_out, labels, indexes, self.feature_memory, k, patch_rate, positive_rate)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, path_list, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), path_list

    def _forward(self, inputs):
        return self.encoder(inputs)

    def UnNormalize(self, tensor):
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                                       transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                            std=[1., 1., 1.]),
                                       ])

        inv_tensor = invTrans(tensor)
        return inv_tensor