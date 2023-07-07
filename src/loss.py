import numpy as np
import torch
import torch.nn as nn


def get_loss(loss_fn: str, temp: float = 0.5):
    """
    :param loss_fn:
    :param temp:
    :return:
    """
    if loss_fn == 'ce':
        return nn.CrossEntropyLoss()

    elif loss_fn == 'sscl':
        """ SimCLR: https://proceedings.mlr.press/v119/chen20j/chen20j.pdf """
        return InfoNCE(temperature=temp)

    elif loss_fn == 'scl':
        return SupConLoss(temperature=temp)

    else:
        raise NotImplementedError


# ------------- Contrastive Losses  ------------------ #
def compute_inner_pdt_mtx(z, z_aug, temp):
    """
    compute Temp normalized - cross similarity (inner product) scores
    o/p [i,j] th entry:  [ exp(z_i, z_j) ]; [i, i] th entry = 0
    """
    z = torch.nn.functional.normalize(z, dim=1)
    z_aug = torch.nn.functional.normalize(z_aug, dim=1)

    # calculate similarities block-wise - the resulting vectors have shape (batch_size, batch_size)
    inner_pdt_00 = torch.einsum('nc,mc->nm', z, z) / temp
    inner_pdt_01 = torch.einsum('nc,mc->nm', z, z_aug) / temp
    inner_pdt_10 = torch.t(inner_pdt_01)
    inner_pdt_11 = torch.einsum('nc,mc->nm', z_aug, z_aug) / temp

    # concatenate blocks : o/p shape (2*batch_size, 2*batch_size)
    # [ Block 00 ] | [ Block 01 ]
    # [ Block 10 ] | [ Block 11 ]
    inner_pdt_0001 = torch.cat([inner_pdt_00, inner_pdt_01], dim=1)
    inner_pdt_1011 = torch.cat([inner_pdt_10, inner_pdt_11], dim=1)
    inner_pdt_mtx = torch.cat([inner_pdt_0001, inner_pdt_1011], dim=0)

    return inner_pdt_mtx


def compute_nll_mtx(inner_pdt_mtx):
    """
    :param inner_pdt_mtx:
    """
    max_inner_pdt, _ = torch.max(inner_pdt_mtx, dim=1, keepdim=True)
    inner_pdt_mtx = inner_pdt_mtx - max_inner_pdt.detach()  # for numerical stability
    nll_mtx = torch.exp(inner_pdt_mtx)
    diag_mask = torch.ones_like(inner_pdt_mtx).fill_diagonal_(0)
    nll_mtx = nll_mtx * diag_mask
    nll_mtx = nll_mtx / torch.sum(nll_mtx, dim=1, keepdim=True)
    nll_mtx[nll_mtx != 0] = - torch.log(nll_mtx[nll_mtx != 0])

    return nll_mtx


def get_self_aug_mask(z):
    """ [i,j] = 1 if x_j is aug of x_i else 0 """
    aug_mask_00 = torch.zeros((z.shape[0], z.shape[0]), device=z.device)
    aug_mask_01 = torch.zeros((z.shape[0], z.shape[0]), device=z.device)
    aug_mask_01.fill_diagonal_(1)
    aug_mask_10 = aug_mask_01
    aug_mask_11 = aug_mask_00
    aug_mask_0001 = torch.cat([aug_mask_00, aug_mask_01], dim=1)
    aug_mask_1011 = torch.cat([aug_mask_10, aug_mask_11], dim=1)
    aug_mask = torch.cat([aug_mask_0001, aug_mask_1011], dim=0)

    neg_aug_mask = aug_mask.clone()
    neg_aug_mask[aug_mask == 0] = 1
    neg_aug_mask[aug_mask == 1] = 0
    neg_aug_mask.fill_diagonal_(0)

    return aug_mask, neg_aug_mask


# -----------------------------------
# Unsupervised losses
# -----------------------------------
class InfoNCE(nn.Module):
    """
        Implements NT-Xent loss
    """

    def __init__(self, temperature: float = 0.5, reduction="mean"):
        super(InfoNCE, self).__init__()
        self.temp = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, z, z_aug, *kwargs):
        """
        :param z:
        :param z_aug:
        :param kwargs:
        :return:
        """
        inner_pdt_mtx = compute_inner_pdt_mtx(z=z, z_aug=z_aug, temp=self.temp)
        bs, _ = z.shape
        labels = torch.arange(bs, device=z.device, dtype=torch.long)
        labels = labels.repeat(2)
        loss = self.cross_entropy(inner_pdt_mtx, labels)
        return loss


class DCL(nn.Module):
    """
    https://arxiv.org/abs/2007.00224
    """

    def __init__(self, temperature: float = 0.5, prior: float = 0.5, reduction: str = 'mean'):
        super(DCL, self).__init__()
        self.temp = temperature
        self.reduction = reduction

        self.pi_p = prior
        self.pi_n = 1 - self.pi_p

        self.tau_p = self.pi_p  # 1 - 2 * self.pi_p * self.pi_n  # prob of two U samples having same label
        self.tau_n = self.pi_n  # 1 - self.tau_p

        self.neg_mask = None
        self.self_aug_mask = None
        self.bs = None
        self.N = None

    def forward(self, z, z_aug, *kwargs):
        """

        :param z:
        :param z_aug:
        :return:
        """
        # calculate: <(z_i^Tz_j/\tau)>
        inner_pdt_mtx = compute_inner_pdt_mtx(z=z, z_aug=z_aug, temp=self.temp)
        # compute < exp(z_i^Tz_j/\tau)>
        similarity_mtx = torch.exp(inner_pdt_mtx)

        # get masks
        if self.bs != z.shape[0] or self.self_aug_mask is None:
            self.bs = z.shape[0]
            self.N = 2 * self.bs - 2
            self.self_aug_mask, self.neg_mask = get_self_aug_mask(z=z)

        # similarity with self_aug ~ R_pp
        pos = (similarity_mtx * self.self_aug_mask).sum(dim=1)
        # similarity with everything else ~ R_un
        neg = (similarity_mtx * self.neg_mask).sum(dim=1)

        # unbiased R_nn = (R_un - \tau_p N R_pp) / tau_n
        Ng = (-self.tau_p * self.N * pos + neg) / self.tau_n
        # clamp at farthest point on the hypersphere for sample i
        Ng = torch.clamp(Ng, min=self.N * np.e ** (-1 / self.temp))

        # estimate infoNCE loss
        debiased_loss = -torch.log(pos / (pos + Ng))

        if self.reduction == 'mean':
            return torch.mean(debiased_loss)

        return debiased_loss


# -----------------------------------
# Supervised losses
# -----------------------------------
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    Attractive force between self augmentation and all other samples from same class
    """

    def __init__(self, temperature: float = 0.5, reduction: str = 'mean'):
        super(SupConLoss, self).__init__()
        self.temp = temperature
        self.reduction = reduction

    def forward(self, z, z_aug, labels=None, *kwargs):
        """

        :param z:
        :param z_aug:
        :param labels:
        :return:
        """
        # compute Temp normalized - cross similarity (inner product) scores
        inner_pdt_mtx = compute_inner_pdt_mtx(z=z, z_aug=z_aug, temp=self.temp)
        similarity_mtx = compute_nll_mtx(inner_pdt_mtx=inner_pdt_mtx)

        # mask out contributions from samples not from same class as i
        mask_label = torch.unsqueeze(labels, dim=-1)
        eq_mask = torch.eq(mask_label, torch.t(mask_label))
        eq_mask = torch.tile(eq_mask, (2, 2))
        similarity_scores = similarity_mtx * eq_mask

        loss = similarity_scores.sum(dim=1) / (eq_mask.sum(dim=1) - 1)

        if self.reduction == 'mean':
            loss = torch.mean(loss)

        return loss


# ----------------------------------------------
# Supervised + Unsupervised Contrastive losses
# ----------------------------------------------

class MixedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    Attractive force between self augmentation and all other samples from same class
    """

    def __init__(self, mixing_wt:float = 0.5, temperature: float = 0.5, reduction: str = 'mean'):
        super(MixedContrastiveLoss, self).__init__()
        self.temp = temperature
        self.reduction = reduction
        self.gamma = mixing_wt
        self.supervised_loss = SupConLoss(temperature = self.temp)
        self.unsupervised_loss = InfoNCE(temperature=self.temp)

    def forward(self, z, z_aug, labels=None, *kwargs):
        """

        :param z:
        :param z_aug:
        :param labels:
        :return:
        """
        # compute Temp normalized - cross similarity (inner product) scores
        unsup_loss = self.unsupervised_loss(z=z, z_aug=z_aug)
        sup_loss = self.supervised_loss(z=z, z_aug=z_aug, labels=labels)

        loss = self.gamma * sup_loss + (1 - self.gamma) * unsup_loss

        return loss
