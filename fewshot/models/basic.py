import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable

from fewshot.models.model_factory import RegisterModel
from fewshot.models.utils import *

import torch.nn.init as nninit
from fewshot.data.episode import Episode
import pdb
import subprocess

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

@RegisterModel("protonet")
class Protonet(nn.Module):

    def __init__(self, config, dataset):
        super(Protonet, self).__init__()

        self.config = config
        self.dataset = dataset

        ##For learning cluster radii
        log_sigma_u = torch.log(torch.FloatTensor([config.init_sigma_u]))
        if config.learn_sigma_u:
            self.log_sigma_u = nn.Parameter(log_sigma_u, requires_grad=True)
        else:
            self.log_sigma_u = Variable(log_sigma_u, requires_grad=True).cuda()

        log_sigma_l = torch.log(torch.FloatTensor([config.init_sigma_l]))
        if config.learn_sigma_l:
            self.log_sigma_l = nn.Parameter(log_sigma_l, requires_grad=True)
        else:
            self.log_sigma_l = Variable(log_sigma_l, requires_grad=True).cuda()

        x_dim = [config.num_channel]
        hid_dim = 64
        z_dim = 64
        use_sigma = True

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        def final_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels+1, 3, padding=1),
                nn.BatchNorm2d(out_channels+1),
                nn.MaxPool2d(2)
            )

        self.encoder = nn.Sequential(
                    conv_block(x_dim[0], hid_dim),
                    conv_block(hid_dim, hid_dim),
                    conv_block(hid_dim, hid_dim),
                    conv_block(hid_dim, z_dim),
                    Flatten()
            )

        self.init_weights()

    def init_weights(self):
        def conv_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)

        self.encoder = self.encoder.apply(conv_init)

    def _compute_distances(self, protos, example):
        dist = torch.sum((example - protos)**2, dim=2)
        return dist

    def _process_batch(self, batch, super_classes=False):
        """Convert np arrays to variable"""
        x_train = Variable(torch.from_numpy(batch.x_train).type(torch.FloatTensor), requires_grad=False).cuda()
        x_test = Variable(torch.from_numpy(batch.x_test).type(torch.FloatTensor), requires_grad=False).cuda()

        if batch.x_unlabel is not None and batch.x_unlabel.size > 0:
            x_unlabel = Variable(torch.from_numpy(batch.x_unlabel).type(torch.FloatTensor), requires_grad=False).cuda()
            y_unlabel = Variable(torch.from_numpy(batch.y_unlabel.astype(np.int64)), requires_grad=False).cuda()
        else:
            x_unlabel = None
            y_unlabel = None

        if super_classes:
            labels_train = Variable(torch.from_numpy(batch.y_train_str[:,1]).type(torch.LongTensor), requires_grad=False).unsqueeze(0).cuda()
            labels_test = Variable(torch.from_numpy(batch.y_test_str[:,1]).type(torch.LongTensor), requires_grad=False).unsqueeze(0).cuda()
        else:
            labels_train = Variable(torch.from_numpy(batch.y_train.astype(np.int64)[:,:,1]),requires_grad=False).cuda()
            labels_test = Variable(torch.from_numpy(batch.y_test.astype(np.int64)[:,:,1]),requires_grad=False).cuda()

        return Episode(x_train,
                                     labels_train,
                                     np.expand_dims(batch.train_indices,0),
                                     x_test,
                                     labels_test,
                                     np.expand_dims(batch.test_indices,0),
                                     x_unlabel=x_unlabel,
                                     y_unlabel=y_unlabel,
                                     unlabel_indices=np.expand_dims(batch.unlabel_indices,0),
                                     y_train_str=batch.y_train_str,
                                     y_test_str=batch.y_test_str)


    def _noisify_labels(self, y_train, num_noisy=1):
        if num_noisy > 0:
            num_classes = len(np.unique(y_train))
            shot = int(y_train.shape[1]/num_classes)
            selected_idxs = y_train[:,::int(shot/num_noisy)]
            y_train[:,::int(shot/num_noisy)] = np.random.randint(0, num_classes, len(selected_idxs[0]))
        return y_train


    def _run_forward(self, cnn_input):
        n_class = cnn_input.size(1)
        n_support = cnn_input.size(0)
        encoded = self.encoder.forward(cnn_input.view(n_class * n_support, *cnn_input.size()[2:]))
        return encoded.unsqueeze(0)

    def _compute_protos(self, h, probs):
        """Compute the prototypes
        Args:
            h: [B, N, D] encoded inputs
            probs: [B, N, nClusters] soft assignment
        Returns:
            cluster protos: [B, nClusters, D]
        """

        h = torch.unsqueeze(h, 2)       # [B, N, 1, D]
        probs = torch.unsqueeze(probs, 3)       # [B, N, nClusters, 1]
        prob_sum = torch.sum(probs, 1)  # [B, nClusters, 1]
        zero_indices = (prob_sum.view(-1) == 0).nonzero()
        if torch.numel(zero_indices) != 0:
            values = torch.masked_select(torch.ones_like(prob_sum), torch.eq(prob_sum, 0.0))
            prob_sum = prob_sum.put_(zero_indices, values)
        protos = h*probs    # [B, N, nClusters, D]
        protos = torch.sum(protos, 1)/prob_sum
        return protos

    def _get_count(self, probs, soft=True):
        """
        Args:
            probs: [B, N, nClusters] soft assignments
        Returns:
            counts: [B, nClusters] number of elements in each cluster
        """
        if not soft:
            _, max_indices = torch.max(probs, 2)    # [B, N]
            nClusters = probs.size()[2]
            max_indices = one_hot(max_indices, nClusters)
            counts = torch.sum(max_indices, 1).cuda()
        else:
            counts = torch.sum(probs, 1)
        return counts

    def _embedding_variance(self, x):
        """Compute variance in embedding space
        Args:
            x: examples from one class
        Returns:
            in-class variance
        """
        h = self._run_forward(x)   # [B, N, D]
        D = h.size()[2]
        h = h.view(-1, D)   # [BxN, D]
        variance = torch.var(h, 0)
        return torch.sum(variance)

    def _within_class_variance(self, x_list):
        protos = []
        for x in x_list:
            h = self._run_forward(x)
            D = h.size()[2]
            h = h.view(-1, D)   # [BxN, D]
            proto = torch.mean(h, 0)
            protos.append(proto)
        protos = torch.cat(protos, 0)
        variance = torch.var(protos, 0)
        return torch.sum(variance)

    def _within_class_distance(self, x_list):
        protos = []
        for x in x_list:
            h = self._run_forward(x)
            D = h.size()[2]
            h = h.view(-1, D)   # [BxN, D]
            proto = torch.mean(h, 0, keepdim=True)
            protos.append(proto)
        protos = torch.cat(protos, 0).data.cpu().numpy()   # [C, D]
        num_classes = protos.shape[0]
        distances = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i > j:
                    dist = np.sum((protos[i, :] - protos[j, :])**2)
                    distances.append(dist)
        return np.mean(distances)


    def forward(self, sample):
        raise NotImplementedError
