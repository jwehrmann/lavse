import torch
import torch.nn as nn
import numpy as np


def adjust_k(epoch, initial_k, increase_k, max_violation=False):
    """
        Update loss hyper-parameter k
        linearly from intial_k to 1 according to
        the number of epochs
    """
    if max_violation:
        return 1.

    return min(initial_k + (increase_k * epoch), 1.)


def cosine_sim(im, s,):
    """
        Cosine similarity between all the
        image and sentence pairs
    """
    return im.mm(s.t())


def cosine_sim_numpy(im, s):
    """
        Cosine similarity between all the
        image and sentence pairs
    """
    return im.dot(s.T)


class ContrastiveLoss_(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(
            self, device,
            margin=0.2, max_violation=True,
            weight=1., initial_k=1., increase_k=0,
        ):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.device = device
        self.sim = cosine_sim
        self.weight = weight
        self.initial_k = initial_k
        self.increase_k = increase_k
        self.max_violation = max_violation
        self.epoch = -1

    def update_epoch(self, epoch=None):
        if epoch is None:
            self.epoch += 1

    def update_k(self,):
        self.k = adjust_k(
            self.epoch,
            initial_k=self.initial_k,
            increase_k=self.increase_k,
            max_violation=self.max_violation,
        )
        return self.k

    def forward(self, scores):

        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask.to(self.device)

        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s_t = cost_s.sum()
        cost_im_t = cost_im.sum()

        k = self.update_k()

        cost_all_k = (cost_s_t + cost_im_t) * (1. - k)

        # keep the maximum violating negative for each query
        cost_s_max = cost_s.max(1)[0]
        cost_im_max = cost_im.max(0)[0]

        cost_hard_k = (cost_s_max.sum() + cost_im_max.sum()) * k

        total_loss = cost_all_k + cost_hard_k

        return total_loss * self.weight



class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(
            self, device,
            margin=0.2, max_violation=True,
            weight=1., beta=0.999,
        ):
        super().__init__()
        self.margin = margin
        self.device = device
        self.sim = cosine_sim
        self.weight = weight
        self.max_violation = max_violation
        self.beta = beta

        self.iteration = 0
        self.k = 0

    def adjust_k(self, ):
        """
            Update loss hyper-parameter k
            linearly from intial_k to 1 according to
            the number of epochs
        """
        if self.max_violation:
            self.k = 1
            return 1.

        self.iteration += 1

        self.k = (1.-self.beta**np.float(self.iteration))
        return self.k

    def forward(self, scores ):
        # compute image-sentence score matrix
        # scores = self.sim(im, s)

        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask.to(self.device)

        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s_t = cost_s.sum()
        cost_im_t = cost_im.sum()

        k = self.adjust_k()

        cost_all_k = (cost_s_t + cost_im_t) * (1. - k)

        # keep the maximum violating negative for each query
        cost_s_max = cost_s.max(1)[0]
        cost_im_max = cost_im.max(0)[0]

        cost_hard_k = (cost_s_max.sum() + cost_im_max.sum()) * k

        total_loss = cost_all_k + cost_hard_k

        return total_loss * self.weight

    def __repr__(self):
        return((
            f'ContrastiveLoss (margin={self.margin}, '
            f'device={self.device}, '
            f'similarity_fn={self.sim}, '
            f'weight={self.weight}, '
            f'max_violation={self.max_violation}, '
            f'beta={self.beta})'
        ))
