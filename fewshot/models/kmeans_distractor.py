from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from fewshot.models.model_factory import RegisterModel
from fewshot.models.imp import IMPModel
from fewshot.models.utils import *
from fewshot.models.weighted_ce_loss import weighted_loss

@RegisterModel("kmeans-distractor")
class KMeansDistractorModel(IMPModel):

    def __init__(self, config, data):
        super(KMeansDistractorModel, self).__init__(config, data)

        self.base_distribution = Variable(0*torch.randn(1, 1, config.dim)).cuda()

    def _add_cluster(self, nClusters, protos, radii, cluster_type='unlabeled'):
        """
        Args:
            nClusters: number of clusters
            protos: [B, nClusters, D] cluster protos
            radii: [B, nClusters] cluster radius
            cluster_type: labeled or unlabeled
        Returns:
            updated arguments
        """
        nClusters += 1
        bsize = protos.size()[0]
        dimension = protos.size()[2]

        new_proto = Variable(self.base_distribution.data).cuda()

        protos = torch.cat([protos, new_proto], dim=1)
        zero_count = Variable(torch.zeros(bsize, 1)).cuda()

        d_radii = Variable(torch.ones(bsize, 1), requires_grad=True).cuda()

        if cluster_type == 'unlabeled':
            d_radii = d_radii * torch.exp(self.log_sigma_u)
        else:
            d_radii = d_radii * torch.exp(self.log_sigma_l)

        radii = torch.cat([radii, d_radii], dim=1)
        return nClusters, protos, radii

    def delete_empty_clusters(self, tensor_proto, prob, radii, eps=1e-6):
        column_sums = torch.sum(prob[0],dim=0).data
        good_protos = column_sums > eps
        idxs = torch.nonzero(good_protos).squeeze()
        return tensor_proto[:,idxs,:], radii[:,idxs]

    def forward(self, sample, super_classes=False):
        batch = self._process_batch(sample, super_classes=super_classes)

        nClusters = len(np.unique(batch.y_train.data.cpu().numpy()))
        nClustersInitial = nClusters

        h_train = self._run_forward(batch.x_train)
        h_test = self._run_forward(batch.x_test)

        prob_train = one_hot(batch.y_train, nClusters).cuda()

        protos = self._compute_protos(h_train, prob_train)

        bsize = h_train.size()[0]

        radii = torch.exp(self.log_sigma_l) * Variable(torch.ones(bsize, nClusters), requires_grad=False).cuda()

        support_labels = torch.arange(0, nClusters).cuda().long()
        unlabeled_flag = torch.LongTensor([-1]).cuda()

        #deal with semi-supervised data
        if batch.x_unlabel is not None:
            h_unlabel = self._run_forward(batch.x_unlabel)
            h_all = torch.cat([h_train, h_unlabel], dim=1)

            #add in distractor cluster centered at zero
            nClusters, protos, radii = self._add_cluster(nClusters, protos, radii, 'unlabeled')
            prob_train = one_hot(batch.y_train, nClusters).cuda()
            support_labels = torch.cat([support_labels, unlabeled_flag], dim=0)

            #perform some clustering steps
            for ii in range(self.config.num_cluster_steps):

                prob_unlabel = assign_cluster_radii(protos, h_unlabel, radii)
                prob_unlabel_nograd = Variable(prob_unlabel.data, requires_grad=False).cuda()

                prob_all = torch.cat([prob_train, prob_unlabel_nograd], dim=1)
                protos = self._compute_protos(h_all, prob_all)


        logits = compute_logits_radii(protos, h_test, radii).squeeze()

        labels = batch.y_test.data
        labels[labels >= nClustersInitial] = -1

        support_targets = labels[0, :, None] == support_labels
        loss = self.loss(logits, support_targets, support_labels)

        # map support predictions back into classes to check accuracy
        _, support_preds = torch.max(logits.data, dim=1)
        y_pred = support_labels[support_preds]

        acc_val = torch.eq(y_pred, labels[0]).float().mean()

        return loss, {
            'loss': loss.data[0],
            'acc': acc_val,
            'logits': logits.data[0]
            }
