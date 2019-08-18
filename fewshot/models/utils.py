import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb

def one_hot(indices, depth, dim=-1, cumulative=True):
    """One-hot encoding along dim"""
    new_size = []
    for ii in range(len(indices.size())):
        if ii == dim:
            new_size.append(depth)
        new_size.append(indices.size()[ii])
    if dim == -1:
        new_size.append(depth)
    
    out = torch.zeros(new_size)
    indices = torch.unsqueeze(indices, dim)
    out = out.scatter_(dim, indices.data.type(torch.LongTensor), 1.0)

    return Variable(out)

def update_params(loss, params_dict, step_size=0.1):
    params= [v for k,v in params_dict.items()]
    updated_params = params_dict.copy()
    grads = torch.autograd.grad(loss, params,
        create_graph=False, allow_unused=True)

    for (name, param), grad in zip(params_dict, grads):
        updated_params[name] = param - step_size * grad
        
    return updated_params 

def get_mean_nonzero(tensor, default):
    if torch.numel(torch.nonzero(tensor)) > 0:
        return tensor.sum() / torch.nonzero(tensor).size(0)
    else:
        return default

def covar_default(data, default=1):
    if data.size(1) > 1:
        cov = data.var(dim=1)
        cov[cov <= 0.001] = 0.001
        return cov
    else:
        return default.expand((1,data.size(2)))

def var_default(data, default=1):
    if data.size(1) > 1:
        return data.var(dim=1).mean()
    else:
        return default

def weighted_var_default(data, weights, default=1):
    if data.size(1) > 1:
        mean = data.mean(dim=1)
        squared = ((data - mean)**2).mean(dim=-1)
        weights_squared = weights**2
        weights_sum = torch.sum(weights)
        result = torch.sum(weights*squared)/torch.sum(weights)
        return weights_sum**2/(weights_sum**2 - torch.sum(weights_squared))*result
    else:
        return default

def reverse_map(y_raw, class_id):
    N = y_raw.size[1]
    out = np.zeros_like(y_raw)
    for i in range(N):
        out = class_id.index(y_raw[0, i])
    return out

def ones_like(variable, requires_grad=False):
    return Variable(torch.ones_like(variable), requires_grad=requires_grad)

def compute_distances(protos, example):
  dist = torch.sum((example - protos)**2, dim=2)
  return dist

def entropy(counts):
    """Compute entropy from discrete counts"""
    if len(counts.shape) > 1:
        counts = counts.flatten()
    N = np.sum(counts)
    p = counts / float(N)
    p = p[np.nonzero(p)]
    return -np.sum(p*np.log(p))


def compute_logits(cluster_centers, data):
    """Computes the logits of being in one cluster, squared Euclidean.
    Args:
        cluster_centers: [B, K, D] Cluster center representation.
        data: [B, N, D] Data representation.
    Returns:
        log_prob: [B, N, K] logits.
    """
    cluster_centers = torch.unsqueeze(cluster_centers, 1)  # [B, 1, K, D]
    data = torch.unsqueeze(data, 2)  # [B, N, 1, D]
    # [B, N, K]
    neg_dist = -torch.sum((data - cluster_centers)**2, 3)
    return neg_dist


def assign_cluster(cluster_centers, data):
    """Assigns data to cluster center, using K-Means.
    Args:
        cluster_centers: [B, K, D] Cluster center representation.
        data: [B, N, D] Data representation.
    Returns:
        prob: [B, N, K] Soft assignment.
    """
    logits = compute_logits(cluster_centers, data)  # [B, N, K]
    logits_shape = logits.size()
    bsize = logits_shape[0]
    ndata = logits_shape[1]
    ncluster = logits_shape[2]
    prob = F.softmax(logits, dim=-1)
    return prob


def assign_cluster_radii(cluster_centers, data, radii):
    """Assigns data to cluster center, using K-Means.

    Args:
        cluster_centers: [B, K, D] Cluster center representation.
        data: [B, N, D] Data representation.
        radii: [B, K] Cluster radii.
    Returns:
        prob: [B, N, K] Soft assignment.
    """
    logits = compute_logits_radii(cluster_centers, data, radii) # [B, N, K]
    logits_shape = logits.size()
    bsize = logits_shape[0]
    ndata = logits_shape[1]
    ncluster = logits_shape[2]
    prob = F.softmax(logits, dim=-1)
    return prob

def assign_cluster_radii_limited(cluster_centers, data, radii, target_labels):
    """Assigns data to cluster center, using K-Means.

    Args:
        cluster_centers: [B, K, D] Cluster center representation.
        data: [B, N, D] Data representation.
        radii: [B, K] Cluster radii.
    Returns:
        prob: [B, N, K] Soft assignment.
    """
    logits = compute_logits_radii(cluster_centers, data, radii) # [B, N, K]
    class_logits = (torch.min(logits).data-100)*torch.ones(logits.data.size()).cuda()
    class_logits[target_labels] = logits.data[target_labels]
    logits_shape = logits.size()
    bsize = logits_shape[0]
    ndata = logits_shape[1]
    ncluster = logits_shape[2]
    prob = F.softmax(Variable(class_logits), dim=-1)
    return prob

def assign_cluster_radii_diag(cluster_centers, data, radii):
    """Assigns data to cluster center, using K-Means.

    Args:
        cluster_centers: [B, K, D] Cluster center representation.
        data: [B, N, D] Data representation.
        radii: [B, K] Cluster radii.
    Returns:
        prob: [B, N, K] Soft assignment.
    """
    logits = compute_logits_radii_diag(cluster_centers, data, radii)    # [B, N, K]
    logits_shape = logits.size()
    bsize = logits_shape[0]
    ndata = logits_shape[1]
    ncluster = logits_shape[2]
    prob = F.softmax(logits, dim=-1)
    return prob

def assign_cluster_crp(cluster_centers, data, priors):
    """Assigns data to cluster center, using K-Means.

    Args:
        cluster_centers: [B, K, D] Cluster center representation.
        data: [B, N, D] Data representation.
        priors: [B, K] Cluster priors.
    Returns:
        prob: [B, N, K] Soft assignment.
    """
    logits = compute_logits_crp(cluster_centers, data, priors) # [B, N, K]
    logits_shape = logits.size()
    bsize = logits_shape[0]
    ndata = logits_shape[1]
    ncluster = logits_shape[2]
    prob = F.softmax(logits, dim=-1)
    return prob


def assign_cluster_radii_crp(cluster_centers, data, radii, priors):
    """Assigns data to cluster center, using K-Means.

    Args:
        cluster_centers: [B, K, D] Cluster center representation.
        data: [B, N, D] Data representation.
        radii: [B, K] Cluster radii.
    Returns:
        prob: [B, N, K] Soft assignment.
    """
    logits = compute_logits_radii_crp(cluster_centers, data, radii, priors)
    logits_shape = logits.size()
    bsize = logits_shape[0]
    ndata = logits_shape[1]
    ncluster = logits_shape[2]
    prob = F.softmax(logits, dim=-1)
    return prob


def compute_logits_radii(cluster_centers, data, radii, prior_weight=1.):
    """Computes the logits of being in one cluster, squared Euclidean.

    Args:
        cluster_centers: [B, K, D] Cluster center representation.
        data: [B, N, D] Data representation.
        radii: [B, K] Cluster radii.
    Returns:
        log_prob: [B, N, K] logits.
    """
    cluster_centers = torch.unsqueeze(cluster_centers, 1)   # [B, 1, K, D]
    data = torch.unsqueeze(data, 2)  # [B, N, 1, D]
    dim = data.size()[-1]
    radii = torch.unsqueeze(radii, 1)  # [B, 1, K]
    neg_dist = -torch.sum((data - cluster_centers)**2, dim=3)   # [B, N, K]

    logits = neg_dist / 2.0 / (radii)
    norm_constant = 0.5*dim*(torch.log(radii) + np.log(2*np.pi))

    logits = logits - norm_constant
    return logits



def compute_logits_crp(cluster_centers, data, priors):
    """Computes the logits of being in one cluster, squared Euclidean.
    Args:
        cluster_centers: [B, K, D] Cluster center representation.
        data: [B, N, D] Data representation.
        priors: [B, K] cluster priors
    Returns:
        log_prob: [B, N, K] logits.
    """
    dimension = cluster_centers.size()[2]
    cluster_centers = torch.unsqueeze(cluster_centers, 1)  # [B, 1, K, D]
    data = torch.unsqueeze(data, 2)  # [B, N, 1, D]
    priors = torch.unsqueeze(priors, 1)   # [B, 1, K]

    neg_dist = -torch.sum((data - cluster_centers)**2, 3)   # [B, N, K] log probs
    log_likelihood = 0.5 * neg_dist #- 0.5 * np.log(2 * np.pi)
    log_posterior = log_likelihood + torch.log(priors)
    return log_posterior


def compute_logits_radii_crp(cluster_centers, data, radii, priors):
    """Computes the logits of being in one cluster, squared Euclidean.

    Args:
        cluster_centers: [B, K, D] Cluster center representation.
        data: [B, N, D] Data representation.
        radii: [B, K] Cluster radii.
    Returns:
        log_prob: [B, N, K] logits.
    """
    dimension = cluster_centers.size()[2]
    cluster_centers = torch.unsqueeze(cluster_centers, 1)  # [B, 1, K, D]
    data = torch.unsqueeze(data, 2)  # [B, N, 1, D]
    dim = data.size()[-1]
    priors = torch.unsqueeze(priors, 1)   # [B, 1, K]  
    radii = torch.unsqueeze(radii, 1)  # [B, 1, K]

    neg_dist = -torch.sum((data - cluster_centers)**2, 3)       # [B, N, K]

    log_likelihood = 0.5 * neg_dist / (radii) - 0.5*dim*(torch.log(radii) - np.log(2 * np.pi))
    log_posterior = log_likelihood + torch.log(priors)
    # print log_likelihood, log_posterior
    return log_posterior

def eval_distractor(pred_non_distractor, gt_non_distractor):
    """Evaluates distractor prediction.

    Args:
        pred_non_distractor
        gt_non_distractor

    Returns:
        acc:
        recall:
        precision:
    """
    y = gt_non_distractor.type(torch.FloatTensor)
    pred_distractor = 1.0 - pred_non_distractor
    non_distractor_correct = torch.eq(pred_non_distractor, y).type(torch.FloatTensor)
    distractor_tp = pred_distractor * (1.0 - y)
    distractor_recall = torch.sum(distractor_tp) / torch.sum(1 - y)
    distractor_precision = torch.sum(distractor_tp) / (
            torch.sum(pred_distractor) +
            torch.eq(torch.sum(pred_distractor), 0.0).type(torch.FloatTensor))
    acc = torch.mean(non_distractor_correct)
    recall = torch.mean(distractor_recall)
    precision = torch.mean(distractor_precision)

    return acc, recall, precision
