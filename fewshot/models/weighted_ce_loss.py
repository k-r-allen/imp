import torch

def log_sum_exp(value, weights, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(weights*torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)
    
def weighted_loss(logits, targets, weights):
	logsumexp = log_sum_exp(logits, weights, dim=1, keepdim=False)
	loss_by_class = -1*class_select(logits,targets) + logsumexp
	return loss_by_class