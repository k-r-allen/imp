import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from fewshot.models.model_factory import RegisterModel
from fewshot.models.imp import IMPModel
from fewshot.models.utils import *

@RegisterModel("soft-nn")
class SoftNN(IMPModel):

    def forward(self, sample, super_classes=False):
        batch = self._process_batch(sample, super_classes=super_classes)

        xs = batch.x_train
        xq = batch.x_test
        y_train_np = batch.y_train.data.cpu().numpy()

        #every data point is its own cluster
        nClusters = len(y_train_np[0])

        h_train = self._run_forward(xs)
        h_test = self._run_forward(xq)

        prob_train = one_hot(Variable(torch.arange(0,nClusters)).unsqueeze(0), nClusters).cuda()
        protos = self._compute_protos(h_train, prob_train)

        bsize = h_train.size()[0]
        radii = Variable(torch.ones(bsize, nClusters)).cuda() * torch.exp(self.log_sigma_l)

        support_labels = batch.y_train.data[0].long()

        logits = compute_logits_radii(protos, h_test, radii).squeeze()

        # convert class targets into indicators for supports in each class
        labels = batch.y_test.data

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