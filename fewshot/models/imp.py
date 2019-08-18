import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from fewshot.models.model_factory import RegisterModel
from fewshot.models.basic import Protonet
from fewshot.models.utils import *
from fewshot.models.weighted_ce_loss import weighted_loss
import pdb

@RegisterModel("imp")
class IMPModel(Protonet):

    def _add_cluster(self, nClusters, protos, radii, cluster_type='unlabeled', ex = None):
        """
        Args:
            nClusters: number of clusters
            protos: [B, nClusters, D] cluster protos
            radii: [B, nClusters] cluster radius,
            cluster_type: ['labeled','unlabeled'], the type of cluster we're adding
            ex: the example to add
        Returns:
            updated arguments
        """
        nClusters += 1
        bsize = protos.size()[0]
        dimension = protos.size()[2]

        zero_count = Variable(torch.zeros(bsize, 1)).cuda()

        d_radii = Variable(torch.ones(bsize, 1), requires_grad=False).cuda()

        if cluster_type == 'labeled':
            d_radii = d_radii * torch.exp(self.log_sigma_l)
        else:
            d_radii = d_radii * torch.exp(self.log_sigma_u)

        if ex is None:
            new_proto = self.base_distribution.data.cuda()
        else:
            new_proto = ex.unsqueeze(0).unsqueeze(0).cuda()

        protos = torch.cat([protos, new_proto], dim=1)
        radii = torch.cat([radii, d_radii], dim=1)
        return nClusters, protos, radii

    def estimate_lambda(self, tensor_proto, semi_supervised):
        # estimate lambda by mean of shared sigmas
        rho = tensor_proto[0].var(dim=0)
        rho = rho.mean()

        if semi_supervised:
            sigma = (torch.exp(self.log_sigma_l).data[0]+torch.exp(self.log_sigma_u).data[0])/2.
        else:
            sigma = torch.exp(self.log_sigma_l).data[0]

        alpha = self.config.ALPHA
        lamda = -2*sigma*np.log(self.config.ALPHA) + self.config.dim*sigma*np.log(1+rho/sigma)

        return lamda

    def delete_empty_clusters(self, tensor_proto, prob, radii, targets, eps=1e-3):
        column_sums = torch.sum(prob[0],dim=0).data
        good_protos = column_sums > eps
        idxs = torch.nonzero(good_protos).squeeze()
        return tensor_proto[:, idxs, :], radii[:, idxs], targets[idxs]

    def loss(self, logits, targets, labels):
        """Loss function to "or" across the prototypes in the class:
        take the loss for the closest prototype in the class and all negatives.
        inputs:
            logits [B, N, nClusters] of nll probs for each cluster
            targets [B, N] of target clusters
        outputs:
            weighted cross entropy such that we have an "or" function
            across prototypes in the class of each query
        """
        targets = targets.cuda()
        # determine index of closest in-class prototype for each query
        target_logits = torch.ones_like(logits.data) * float('-Inf')
        target_logits[targets] = logits.data[targets]
        _, best_targets = torch.max(target_logits, dim=1)
        # mask out everything...
        weights = torch.zeros_like(logits.data)
        # ...then include the closest prototype in each class and unlabeled)
        unique_labels = np.unique(labels.cpu().numpy())
        for l in unique_labels:
            class_mask = labels == l
            class_logits = torch.ones_like(logits.data) * float('-Inf')
            class_logits[class_mask.repeat(logits.size(0), 1)] = logits[class_mask].data.view(logits.size(0), -1)
            _, best_in_class = torch.max(class_logits, dim=1)
            weights[range(0, targets.size(0)), best_in_class] = 1.
        loss = weighted_loss(logits, Variable(best_targets), Variable(weights))
        return loss.mean()

    def forward(self, sample, super_classes=False):
        
        batch = self._process_batch(sample, super_classes=super_classes)
        nClusters = len(np.unique(batch.y_train.data.cpu().numpy()))
        nInitialClusters = nClusters

        #run data through network
        h_train = self._run_forward(batch.x_train)
        h_test = self._run_forward(batch.x_test)

        #create probabilities for points
        _, idx = np.unique(batch.y_train.squeeze().data.cpu().numpy(), return_inverse=True)
        prob_train = one_hot(batch.y_train, nClusters).cuda()

        #make initial radii for labeled clusters
        bsize = h_train.size()[0]
        radii = Variable(torch.ones(bsize, nClusters)).cuda() * torch.exp(self.log_sigma_l)

        support_labels = torch.arange(0, nClusters).cuda().long()

        #compute initial prototypes from labeled examples
        protos = self._compute_protos(h_train, prob_train)
        
        #estimate lamda
        lamda = self.estimate_lambda(protos.data, batch.x_unlabel is not None)

        #loop for a given number of clustering steps
        for ii in range(self.config.num_cluster_steps):
            tensor_proto = protos.data
            #iterate over labeled examples to reassign first
            for i, ex in enumerate(h_train[0]):
                idxs = torch.nonzero(batch.y_train.data[0, i] == support_labels)[0]
                distances = self._compute_distances(tensor_proto[:, idxs, :], ex.data)
                if (torch.min(distances) > lamda):
                    nClusters, tensor_proto, radii  = self._add_cluster(nClusters, tensor_proto, radii, cluster_type='labeled', ex=ex.data)
                    support_labels = torch.cat([support_labels, batch.y_train[0, i].data], dim=0)

            #perform partial reassignment based on newly created labeled clusters
            if nClusters > nInitialClusters:
                support_targets = batch.y_train.data[0, :, None] == support_labels
                prob_train = assign_cluster_radii_limited(Variable(tensor_proto), h_train, radii, support_targets)

            nTrainClusters = nClusters

            #iterate over unlabeled examples
            if batch.x_unlabel is not None:
                h_unlabel = self._run_forward(batch.x_unlabel)
                h_all = torch.cat([h_train, h_unlabel], dim=1)
                unlabeled_flag = torch.LongTensor([-1]).cuda()

                for i, ex in enumerate(h_unlabel[0]):
                    distances = self._compute_distances(tensor_proto, ex.data)
                    if torch.min(distances) > lamda:
                        nClusters, tensor_proto, radii = self._add_cluster(nClusters, tensor_proto, radii, cluster_type='unlabeled', ex=ex.data)
                        support_labels = torch.cat([support_labels, unlabeled_flag], dim=0)

                # add new, unlabeled clusters to the total probability
                if nClusters > nTrainClusters:
                    unlabeled_clusters = torch.zeros(prob_train.size(0), prob_train.size(1), nClusters - nTrainClusters)
                    prob_train = torch.cat([prob_train, Variable(unlabeled_clusters).cuda()], dim=2)

                prob_unlabel = assign_cluster_radii(Variable(tensor_proto).cuda(), h_unlabel, radii)
                prob_unlabel_nograd = Variable(prob_unlabel.data, requires_grad=False).cuda()
                prob_all = torch.cat([Variable(prob_train.data, requires_grad=False), prob_unlabel_nograd], dim=1)

                protos = self._compute_protos(h_all, prob_all)
                protos, radii, support_labels = self.delete_empty_clusters(protos, prob_all, radii, support_labels)
            else:
                protos = Variable(tensor_proto).cuda()
                protos = self._compute_protos(h_train, Variable(prob_train.data, requires_grad=False).cuda())
                protos, radii, support_labels = self.delete_empty_clusters(protos, prob_train, radii, support_labels)

        logits = compute_logits_radii(protos, h_test, radii).squeeze()

        # convert class targets into indicators for supports in each class
        labels = batch.y_test.data
        labels[labels >= nInitialClusters] = -1

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

    def forward_unsupervised(self, sample, super_classes, unlabel_lambda=20., num_cluster_steps=5):

        batch = self._process_batch(sample, super_classes=super_classes)

        xq = batch.x_test

        h_test = self._run_forward(xq)

        if batch.x_unlabel is not None:
            h_unlabel = self._run_forward(batch.x_unlabel)
            h_all = h_unlabel
            protos = h_unlabel[0][0].unsqueeze(0).unsqueeze(0)

            radii = Variable(torch.ones(1, 1)).cuda() * torch.exp(self.log_sigma_l)
            nClusters = 1
            protos_clusters = []

            for ii in range(num_cluster_steps):
                tensor_proto = protos.data
                for i, ex in enumerate(h_unlabel[0]):
                    distances = self._compute_distances(tensor_proto, ex.data)
                    if (torch.min(distances) > unlabel_lambda):
                        nClusters, tensor_proto, radii = self._add_cluster(nClusters, tensor_proto, radii, 'labeled', ex.data)

                prob_unlabel = assign_cluster_radii(Variable(tensor_proto).cuda(), h_unlabel, radii)

                prob_unlabel_nograd = Variable(prob_unlabel.data, requires_grad=False).cuda()
                prob_all = prob_unlabel_nograd
                protos = self._compute_protos(h_all, prob_all)

        return {'logits':prob_unlabel_nograd.data[0]}