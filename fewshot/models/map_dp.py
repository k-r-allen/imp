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
import pdb

@RegisterModel("map-dp")
class MapDPModel(IMPModel):

    def __init__(self, config, data):
        super(MapDPModel, self).__init__(config, data)

        self.base_distribution = Variable(torch.zeros(1,1,config.dim)).cuda()

    def _compute_priors(self, counts):
        """
        Args:
            counts: [B, nClusters] number of elements in each cluster
        Returns:
            priors: [B, nClusters] crp prior
        """
        ALPHA = self.config.ALPHA
        nClusters = counts.size()[1]
        crp_prior_old = counts
        crp_prior_new = Variable(torch.FloatTensor([ALPHA])).cuda()
        crp_prior_new = torch.cat([crp_prior_new.unsqueeze(0)]*nClusters, 1)
        indices = counts.view(-1).nonzero()
        if torch.numel(indices) != 0:
            values = torch.masked_select(crp_prior_old, torch.gt(counts, 0.0))
            crp_priors = crp_prior_new.put_(indices, values) #if going for uniform, can switch values to torch.ones_like(values)
        else:
            crp_priors = crp_prior_new

        return crp_priors

    def _update_hypers(self, h, total_prob, counts,mu0):
        sigmas = (1./(1./torch.exp(self.log_sigma_u)+1./torch.exp(self.log_sigma_l)*counts))
        revised_means = sigmas.unsqueeze(-1)*(counts.unsqueeze(-1)*self._compute_protos(h, total_prob)/torch.exp(self.log_sigma_l) + mu0/torch.exp(self.log_sigma_u))
        sigmas_out = sigmas + torch.exp(self.log_sigma_l)
        return sigmas_out, revised_means

    def _add_cluster(self, nClusters, protos, counts, radii, mu0):
        """
        Args:
            nClusters: number of clusters
            protos: [B, nClusters, D] cluster protos
            counts: [B, nClusters] number of examples in each cluster
            radii: [B, nClusters] cluster radius,
            cluster_type: ['labeled','unlabeled'], the type of cluster we're adding
        Returns:
            updated arguments
        """
        nClusters += 1
        bsize = protos.size()[0]
        dimension = protos.size()[2]

        zero_count = Variable(torch.zeros(bsize, 1)).cuda()
        counts = torch.cat([counts, zero_count], dim=1)

        d_radii = Variable(torch.ones(bsize, 1), requires_grad=False).cuda()
        d_radii = d_radii*(torch.exp(self.log_sigma_u)+torch.exp(self.log_sigma_l))

        new_proto = mu0.clone()

        protos = torch.cat([protos, new_proto], dim=1)
        radii = torch.cat([radii, d_radii], dim=1)
        return nClusters, protos, counts, radii

    def forward(self, sample, super_classes=False):
        batch = self._process_batch(sample, super_classes=super_classes)

        xs = batch.x_train
        xq = batch.x_test

        y_train_np = batch.y_train.data.cpu().numpy()

        nClusters = len(np.unique(y_train_np))
        shot = int(batch.y_train.shape[1]/nClusters)

        nInitialClusters = nClusters
        h_train = self._run_forward(xs)
        h_test = self._run_forward(xq)

        prob_train = one_hot(batch.y_train, nClusters).cuda()

        if batch.x_unlabel is None:
            protos_clusters = [self._compute_protos(h_train, prob_train)]

        bsize = h_train.size()[0]
        radii = Variable(torch.ones(bsize, nClusters)).cuda() * torch.exp(self.log_sigma_l)

        target_labels = torch.arange(0, nClusters).long()
        protos = self._compute_protos(h_train[:,:,:], prob_train[:,:,:])

        if batch.x_unlabel is not None:
            h_unlabel = self._run_forward(batch.x_unlabel)
            h_all = torch.cat([h_train, h_unlabel],dim=1)
            total_prob = prob_train
        else:
            h_all = h_train
            total_prob = prob_train

        mu0 = torch.mean(protos,dim=1).unsqueeze(0)
        counts = self._get_count(total_prob, soft=False)


        for ii in range(self.config.num_cluster_steps):
            if ii == 0:
                nClusters, protos, counts, radii  = self._add_cluster(nClusters, protos, counts, radii,mu0)
                total_prob = torch.cat([total_prob, Variable(torch.zeros(total_prob.size()[0], total_prob.size()[1], 1)).cuda()],dim=2)
                target_labels = torch.cat([target_labels, torch.LongTensor([-1])],dim=0)

            if batch.x_unlabel is not None:
                for i, ex in enumerate(h_unlabel[0]):
                    counts = self._get_count(total_prob, soft=False)
                    priors = self._compute_priors(counts)
                    radii, protos = self._update_hypers(h_all[:,:total_prob.size()[1],:], total_prob, counts, mu0)
                    cluster_prob = assign_cluster_radii_crp(protos, ex.unsqueeze(0).unsqueeze(0), radii, priors)

                    _, max_val = torch.max(cluster_prob,dim=-1)

                    if i+h_train.size()[1]>=total_prob.size()[1]:
                        total_prob = torch.cat([total_prob, one_hot(max_val, nClusters).cuda()],dim=1)
                    else:
                        total_prob[0,i+h_train.size()[1],:] = one_hot(max_val, nClusters).cuda()

                    if max_val[0].data[0] == nClusters:
                        nClusters, protos, counts, radii  = self._add_cluster(nClusters, protos, counts, radii, mu0)
                        target_labels = torch.cat([target_labels, torch.LongTensor([-1])],dim=0)
                        total_prob = torch.cat([total_prob, Variable(torch.zeros(total_prob.size()[0], total_prob.size()[1], 1)).cuda()],dim=2)
                    

        final_prob = Variable(total_prob.data, requires_grad=False)
        counts = self._get_count(final_prob, soft=False)
        priors = self._compute_priors(counts)
        radii, protos = self._update_hypers(h_all, final_prob, counts, mu0)                

        logits = compute_logits_radii_crp(protos, h_test, radii, priors).squeeze()

        # convert class targets into indicators for supports in each class
        labels = batch.y_test.data
        labels[labels >= nInitialClusters] = -1

        support_targets = labels[0, :, None] == target_labels.cuda()
        loss = self.loss(logits, support_targets, target_labels.cuda())

        # map support predictions back into classes to check accuracy
        _, support_preds = torch.max(logits.data, dim=1)
        y_pred = target_labels.cuda()[support_preds]

        acc_val = torch.eq(y_pred, labels[0]).float().mean()

        return loss, {
            'loss': loss.data[0],
            'acc': acc_val,
            'logits': logits.data[0]
            }
