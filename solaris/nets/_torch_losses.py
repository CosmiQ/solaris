import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch import nn
try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


class TorchFocalLoss(nn.Module):
    """Implementation of Focal Loss[1]_ modified from Catalyst [2]_ .

    Arguments
    ---------
    gamma : :class:`int` or :class:`float`
        Focusing parameter. See [1]_ .
    alpha : :class:`int` or :class:`float`
        Normalization factor. See [1]_ .

    References
    ----------
    .. [1] https://arxiv.org/pdf/1708.02002.pdf
    .. [2] https://catalyst-team.github.io/catalyst/
    """

    def __init__(self, gamma=2, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    # TODO refactor
    def forward(self, outputs, targets):
        """Calculate the loss function between `outputs` and `targets`.

        Arguments
        ---------
        outputs : :class:`torch.Tensor`
            The output tensor from a model.
        targets : :class:`torch.Tensor`
            The training target.

        Returns
        -------
        loss : :class:`torch.Variable`
            The loss value.
        """
        if targets.size() != outputs.size():
            raise ValueError(
                f"Targets and inputs must be same size. "
                f"Got ({targets.size()}) and ({outputs.size()})"
            )

        max_val = (-outputs).clamp(min=0)
        log_ = ((-max_val).exp() + (-outputs - max_val).exp()).log()
        loss = outputs - outputs * targets + max_val + log_

        invprobs = F.logsigmoid(-outputs * (targets * 2.0 - 1.0))
        loss = self.alpha*(invprobs * self.gamma).exp() * loss

        return loss.sum(dim=-1).mean()


def torch_lovasz_hinge(logits, labels, per_image=False, ignore=None):
    """Lovasz Hinge Loss. Implementation edited from Maxim Berman's GitHub.

    References
    ----------
    https://github.com/bermanmaxim/LovaszSoftmax/
    https://arxiv.org/abs/1705.08790

    Arguments
    ---------
    logits: :class:`torch.Variable`
        logits at each pixel (between -inf and +inf)
    labels: :class:`torch.Tensor`
        binary ground truth masks (0 or 1)
    per_image: bool, optional
        compute the loss per image instead of per batch. Defaults to ``False``.
    ignore: optional void class id.

    Returns
    -------
    loss : :class:`torch.Variable`
        Lovasz loss value for the input logits and labels. Compatible with
        ``loss.backward()`` as its a :class:`torch.Variable` .
    """
    # TODO: Restructure into a class like TorchFocalLoss for compatibility
    if per_image:
        loss = mean(
            lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0),
                                                     lab.unsqueeze(0),
                                                     ignore))
            for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits,
                                                        labels,
                                                        ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss.

    Arguments
    ---------
    logits: :class:`torch.Variable`
        Logits at each prediction (between -inf and +inf)
    labels: :class:`torch.Tensor`
        binary ground truth labels (0 or 1)

    Returns
    -------
    loss : :class:`torch.Variable`
        Lovasz loss value for the input logits and labels.
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class TorchStableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(TorchStableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -inf and +inf)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = TorchStableBCELoss()(logits, Variable(labels.float()))
    return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1 - pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean across images if per_image
    return 100 * np.array(ious)


# helper functions
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


torch_losses = {
    'l1loss': nn.L1Loss,
    'l1': nn.L1Loss,
    'mae': nn.L1Loss,
    'mean_absolute_error': nn.L1Loss,
    'smoothl1loss': nn.SmoothL1Loss,
    'smoothl1': nn.SmoothL1Loss,
    'mean_squared_error': nn.MSELoss,
    'mse': nn.MSELoss,
    'mseloss': nn.MSELoss,
    'categorical_crossentropy': nn.CrossEntropyLoss,
    'cce': nn.CrossEntropyLoss,
    'crossentropyloss': nn.CrossEntropyLoss,
    'negative_log_likelihood': nn.NLLLoss,
    'nll': nn.NLLLoss,
    'nllloss': nn.NLLLoss,
    'poisson_negative_log_likelihood': nn.PoissonNLLLoss,
    'poisson_nll': nn.PoissonNLLLoss,
    'poissonnll': nn.PoissonNLLLoss,
    'kullback_leibler_divergence': nn.KLDivLoss,
    'kld': nn.KLDivLoss,
    'kldivloss': nn.KLDivLoss,
    'binary_crossentropy': nn.BCELoss,
    'bce': nn.BCELoss,
    'bceloss': nn.BCELoss,
    'hinge': nn.HingeEmbeddingLoss,
    'hingeembeddingloss': nn.HingeEmbeddingLoss,
    'multiclass_hinge': nn.MultiMarginLoss,
    'multimarginloss': nn.MultiMarginLoss,
    'softmarginloss': nn.SoftMarginLoss,
    'softmargin': nn.SoftMarginLoss,
    'multiclass_softmargin': nn.MultiLabelSoftMarginLoss,
    'multilabelsoftmarginloss': nn.MultiLabelSoftMarginLoss,
    'cosine': nn.CosineEmbeddingLoss,
    'cosineloss': nn.CosineEmbeddingLoss,
    'cosineembeddingloss': nn.CosineEmbeddingLoss,
    'lovaszhinge': torch_lovasz_hinge,
    'focalloss': TorchFocalLoss,
    'focal': TorchFocalLoss
}
