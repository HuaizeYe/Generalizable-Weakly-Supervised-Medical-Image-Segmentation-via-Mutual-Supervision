import math
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import utils.functional_tensor as functional_tensor

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss_multi(score, target, num_classes, ignore_index=255):
    target = target.float()
    smooth = 1e-5
    loss = 0
    count = 0
    for i in range(num_classes):
        if i is ignore_index:
            continue
        count += 1
        intersect = torch.sum(score[:, i, ...] * (target == i))
        y_sum = torch.sum((target == i) * (target == i))
        z_sum = torch.sum(score[:, i, ...] * score[:, i, ...])
        loss_i = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss_i = 1 - loss_i
        loss = loss + loss_i
    return loss / count

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


class CrossEntropyLoss():
    def __init__(self, mode='all', epsilon=1e-6):
        self.mode = mode
        self.epsilon = epsilon

    def __call__(self, ypred, ytrue):
        ypred = torch.clamp(ypred, self.epsilon, 1 - self.epsilon)
        loss_pos = -ytrue * torch.log(ypred)
        loss_neg = -(1 - ytrue) * torch.log(1 - ypred)

        loss_pos = torch.sum(loss_pos, dim=(0, 2, 3))
        loss_neg = torch.sum(loss_neg, dim=(0, 2, 3))
        nb_pos = torch.sum(ytrue, dim=(0, 2, 3))
        nb_neg = torch.sum(1 - ytrue, dim=(0, 2, 3))

        if self.mode == 'all':
            loss = (loss_pos + loss_neg) / (nb_pos + nb_neg)
        elif self.mode == 'balance':
            loss = (loss_pos / nb_pos + loss_neg / nb_neg) / 2
        return loss


class MILUnarySigmoidLoss():
    def __init__(self, mode='all', focal_params={'alpha': 0.25, 'gamma': 2.0, 'sampling_prob': 1.0}, epsilon=1e-6):
        super(MILUnarySigmoidLoss, self).__init__()
        self.mode = mode
        self.focal_params = focal_params
        self.epsilon = epsilon

    def __call__(self, ypred, mask, gt_boxes):
        loss = mil_unary_sigmoid_loss(ypred, mask, gt_boxes, mode=self.mode,
                                      focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


class MILUnaryApproxSigmoidLoss():
    def __init__(self, mode='all', method='gm', gpower=4,
                 focal_params={'alpha': 0.25, 'gamma': 2.0, 'sampling_prob': 1.0}, epsilon=1e-6):
        super(MILUnaryApproxSigmoidLoss, self).__init__()
        self.mode = mode
        self.method = method
        self.gpower = gpower
        self.focal_params = focal_params
        self.epsilon = epsilon

    def __call__(self, ypred, mask, gt_boxes):
        loss = mil_unary_approx_sigmoid_loss(ypred, mask, gt_boxes, mode=self.mode,
                                             method=self.method, gpower=self.gpower,
                                             focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


class MILUnaryParallelSigmoidLoss():
    def __init__(self, mode='all', angle_params=(-45, 46, 5),
                 focal_params={'alpha': 0.25, 'gamma': 2.0, 'sampling_prob': 1.0},
                 obj_size=0, epsilon=1e-6):
        super(MILUnaryParallelSigmoidLoss, self).__init__()
        self.mode = mode
        self.angle_params = angle_params
        self.focal_params = focal_params
        self.obj_size = obj_size
        self.epsilon = epsilon

    def __call__(self, ypred, mask, crop_boxes):
        loss = mil_parallel_unary_sigmoid_loss(ypred, mask, crop_boxes, angle_params=self.angle_params,
                                               mode=self.mode, focal_params=self.focal_params,
                                               obj_size=self.obj_size, epsilon=self.epsilon)
        return loss


class MILUnaryParallelApproxSigmoidLoss():
    def __init__(self, mode='all', angle_params=(-45, 46, 5), method='gm', gpower=4,
                 focal_params={'alpha': 0.25, 'gamma': 2.0, 'sampling_prob': 1.0},
                 obj_size=0, epsilon=1e-6):
        super(MILUnaryParallelApproxSigmoidLoss, self).__init__()
        self.mode = mode
        self.angle_params = angle_params
        self.method = method
        self.gpower = gpower
        self.focal_params = focal_params
        self.obj_size = obj_size
        self.epsilon = epsilon

    def __call__(self, ypred, mask, crop_boxes):
        loss = mil_parallel_approx_sigmoid_loss(ypred, mask, crop_boxes, angle_params=self.angle_params,
                                                mode=self.mode, focal_params=self.focal_params,
                                                obj_size=self.obj_size, epsilon=self.epsilon)
        return loss


class MILPairwiseLoss():
    def __init__(self, softmax=True, exp_coef=-1):
        super(MILPairwiseLoss, self).__init__()
        self.softmax = softmax
        self.exp_coef = exp_coef

    def __call__(self, ypred, mask):
        loss = mil_pairwise_loss(ypred, mask, softmax=self.softmax, exp_coef=self.exp_coef)
        return loss


def mil_unary_sigmoid_loss(ypred, mask, gt_boxes, mode='all',
                           focal_params={'alpha': 0.25, 'gamma': 2.0, 'sampling_prob': 1.0},
                           epsilon=1e-6):
    """ Compute the mil unary loss.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes: Tensor of boxes with (N, 6), where N is the number of bouding boxes in the batch,
                    the 6 elements of each row are [nb_img, class, x1, y1, x2, y2]
    Returns
        unary loss for each category (C,) if mode='balance'
        otherwise, the average unary loss (1,) if mode='all'
    """
    assert (mode == 'all') | (mode == 'balance') | (mode == 'focal') | (mode == 'mil_focal')
    ypred = torch.clamp(ypred, epsilon, 1 - epsilon)
    num_classes = ypred.shape[1]
    ypred_pos = {c: [] for c in range(num_classes)}
    for nb_ob in range(gt_boxes.shape[0]):
        nb_img = gt_boxes[nb_ob, 0]
        c = gt_boxes[nb_ob, 1].item()
        box = gt_boxes[nb_ob, 2:]
        pred = ypred[nb_img, c, box[1]:box[3] + 1, box[0]:box[2] + 1]
        # print('***',c,box, pred.shape)
        if pred.numel() == 0:
            continue
        ypred_pos[c].append(torch.max(pred, dim=0)[0])
        ypred_pos[c].append(torch.max(pred, dim=1)[0])

    if mode == 'focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask  # weights for negative samples
        weight = weight * (torch.rand(ypred.shape, dtype=ypred.dtype, device=ypred.device) < sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:, c, :, :]
            y_neg = y_neg[(mask[:, c, :, :] < 0.5) & (weight[:, c, :, :] > 0.5)]
            bce_neg = -(1 - alpha) * (y_neg ** gamma) * torch.log(1 - y_neg)
            if len(ypred_pos[c]) > 0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha * ((1 - y_pos) ** gamma) * torch.log(y_pos)
                loss = (bce_neg.sum() + bce_pos.sum()) / len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode == 'mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        v1 = torch.max(ypred * (1 - mask), dim=2)[0]
        v2 = torch.max(ypred * (1 - mask), dim=3)[0]
        ypred_neg = torch.cat([v1, v2], dim=-1).permute(1, 0, 2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0], -1))

        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1 - alpha) * (ypred_neg ** gamma) * torch.log(1 - ypred_neg)
            if len(ypred_pos[c]) > 0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha * ((1 - y_pos) ** gamma) * torch.log(y_pos)
                loss = (bce_neg.sum() + bce_pos.sum()) / len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        ## for negative class
        v1 = torch.max(ypred * (1 - mask), dim=2)[0]
        v2 = torch.max(ypred * (1 - mask), dim=3)[0]
        ypred_neg = torch.cat([v1, v2], dim=-1).permute(1, 0, 2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0], -1))

        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1 - ypred_neg[c])
            if len(ypred_pos[c]) > 0:
                bce_pos = -torch.log(torch.cat(ypred_pos[c], dim=0))
                if mode == 'all':
                    loss = (bce_pos.sum() + bce_neg.sum()) / (len(bce_pos) + len(bce_neg))
                elif mode == 'balance':
                    loss = (bce_pos.mean() + bce_neg.mean()) / 2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    return losses


def mil_unary_approx_sigmoid_loss(ypred, mask, gt_boxes, mode='all', method='gm', gpower=4,
                                  focal_params={'alpha': 0.25, 'gamma': 2.0, 'sampling_prob': 1.0}, epsilon=1e-6):
    assert (mode == 'all') | (mode == 'balance') | (mode == 'focal') | (mode == 'mil_focal')
    ypred = torch.clamp(ypred, epsilon, 1 - epsilon)
    if method == 'gm':
        ypred_g = ypred ** gpower
    elif method == 'expsumr':  # alpha-softmax function
        ypred_g = torch.exp(gpower * ypred)
    elif method == 'explogs':  # alpha-quasimax function
        ypred_g = torch.exp(gpower * ypred)
    num_classes = ypred.shape[1]
    ypred_pos = {c: [] for c in range(num_classes)}
    for nb_ob in range(gt_boxes.shape[0]):
        nb_img = gt_boxes[nb_ob, 0]
        c = gt_boxes[nb_ob, 1].item()
        box = gt_boxes[nb_ob, 2:]
        pred = ypred_g[nb_img, c, box[1]:box[3] + 1, box[0]:box[2] + 1]
        if method == 'gm':
            prob0 = torch.mean(pred, dim=0) ** (1.0 / gpower)
            prob1 = torch.mean(pred, dim=1) ** (1.0 / gpower)
        elif method == 'expsumr':
            pd_org = ypred[nb_img, c, box[1]:box[3] + 1, box[0]:box[2] + 1]
            prob0 = torch.sum(pd_org * pred, dim=0) / torch.sum(pred, dim=0)
            prob1 = torch.sum(pd_org * pred, dim=1) / torch.sum(pred, dim=1)
        elif method == 'explogs':
            msk = mask[nb_img, c, box[1]:box[3] + 1, box[0]:box[2] + 1]
            prob0 = torch.log(torch.sum(pred, dim=0)) / gpower - torch.log(torch.sum(msk, dim=0)) / gpower
            prob1 = torch.log(torch.sum(pred, dim=1)) / gpower - torch.log(torch.sum(msk, dim=1)) / gpower
        ypred_pos[c].append(prob0)
        ypred_pos[c].append(prob1)

    if mode == 'focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask  # weights for negative samples
        weight = weight * (torch.rand(ypred.shape, dtype=ypred.dtype, device=ypred.device) < sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:, c, :, :]
            y_neg = y_neg[(mask[:, c, :, :] < 0.5) & (weight[:, c, :, :] > 0.5)]
            bce_neg = -(1 - alpha) * (y_neg ** gamma) * torch.log(1 - y_neg)
            if len(ypred_pos[c]) > 0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1 - epsilon)
                bce_pos = -alpha * ((1 - y_pos) ** gamma) * torch.log(y_pos)
                loss = (bce_neg.sum() + bce_pos.sum()) / len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode == 'mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        ## for negative class
        if method == 'gm':
            v1 = (torch.sum(ypred_g * (1 - mask), dim=2) / torch.sum(1 - mask, dim=2)) ** (1.0 / gpower)
            v2 = (torch.sum(ypred_g * (1 - mask), dim=3) / torch.sum(1 - mask, dim=3)) ** (1.0 / gpower)
        elif method == 'expsumr':
            v1 = torch.sum(ypred * ypred_g * (1 - mask), dim=2) / torch.sum(ypred_g * (1 - mask), dim=2)
            v2 = torch.sum(ypred * ypred_g * (1 - mask), dim=3) / torch.sum(ypred_g * (1 - mask), dim=3)
        elif method == 'explogs':
            v1 = torch.log(torch.sum(ypred_g * (1 - mask), dim=2)) / gpower - torch.log(
                torch.sum(1 - mask, dim=2)) / gpower
            v2 = torch.log(torch.sum(ypred_g * (1 - mask), dim=3)) / gpower - torch.log(
                torch.sum(1 - mask, dim=3)) / gpower
        ypred_neg = torch.cat([v1, v2], dim=-1).permute(1, 0, 2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0], -1))

        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1 - alpha) * (ypred_neg ** gamma) * torch.log(1 - ypred_neg)
            if len(ypred_pos[c]) > 0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha * ((1 - y_pos) ** gamma) * torch.log(y_pos)
                loss = (bce_neg.sum() + bce_pos.sum()) / len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        ## for negative class
        if method == 'gm':
            v1 = (torch.sum(ypred_g * (1 - mask), dim=2) / torch.sum(1 - mask, dim=2)) ** (1.0 / gpower)
            v2 = (torch.sum(ypred_g * (1 - mask), dim=3) / torch.sum(1 - mask, dim=3)) ** (1.0 / gpower)
        elif method == 'expsumr':
            v1 = torch.sum(ypred * ypred_g * (1 - mask), dim=2) / torch.sum(ypred_g * (1 - mask), dim=2)
            v2 = torch.sum(ypred * ypred_g * (1 - mask), dim=3) / torch.sum(ypred_g * (1 - mask), dim=3)
        elif method == 'explogs':
            v1 = torch.log(torch.sum(ypred_g * (1 - mask), dim=2)) / gpower - torch.log(
                torch.sum(1 - mask, dim=2)) / gpower
            v2 = torch.log(torch.sum(ypred_g * (1 - mask), dim=3)) / gpower - torch.log(
                torch.sum(1 - mask, dim=3)) / gpower
        ypred_neg = torch.cat([v1, v2], dim=-1).permute(1, 0, 2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0], -1))
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            neg = torch.clamp(1 - ypred_neg[c], epsilon, 1 - epsilon)
            bce_neg = -torch.log(neg)
            if len(ypred_pos[c]) > 0:
                pos = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1 - epsilon)
                bce_pos = -torch.log(pos)
                if mode == 'all':
                    loss = (bce_pos.sum() + bce_neg.sum()) / (len(bce_pos) + len(bce_neg))
                elif mode == 'balance':
                    loss = (bce_pos.mean() + bce_neg.mean()) / 2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    return losses


def mil_parallel_unary_sigmoid_loss(ypred, mask, crop_boxes, angle_params=(0, 45, 5), mode='all',
                                    focal_params={'alpha': 0.25, 'gamma': 2.0, 'sampling_prob': 1.0},
                                    obj_size=0, epsilon=1e-6):
    """ Compute the mil unary loss from parallel transformation.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        crop_boxes: Tensor of boxes with (N, 5), where N is the number of bouding boxes in the batch,
                    the 5 elements of each row are [nb_img, class, center_x, center_r, radius]
    Returns
        polar unary loss for each category (C,) if mode='balance'
        otherwise, the average polar unary loss (1,) if mode='all'
    """
    assert (mode == 'all') | (mode == 'balance') | (mode == 'focal') | (mode == 'mil_focal')
    ypred = torch.clamp(ypred, epsilon, 1 - epsilon)
    num_classes = ypred.shape[1]
    ob_img_index = crop_boxes[:, 0].type(torch.int32)
    ob_class_index = crop_boxes[:, 1].type(torch.int32)
    ob_crop_boxes = crop_boxes[:, 2:]
    ypred_pos = {c: [] for c in range(num_classes)}
    for nb_ob in range(crop_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c = ob_class_index[nb_ob]
        radius = ob_crop_boxes[nb_ob, -1]

        extra = 5
        cx, cy, r = ob_crop_boxes[nb_ob, :].type(torch.int32)
        r = r + extra
        xmin = torch.clamp(cx - r, 0)
        ymin = torch.clamp(cy - r, 0)
        pred = ypred[nb_img, c, ymin:cy + r + 1, xmin:cx + r + 1][None, :, :]
        msk = mask[nb_img, c, ymin:cy + r + 1, xmin:cx + r + 1][None, :, :]

        index = torch.nonzero(msk[0] > 0.5, as_tuple=True)
        y0, y1 = index[0].min(), index[0].max()
        x0, x1 = index[1].min(), index[1].max()
        box_h = y1 - y0 + 1
        box_w = x1 - x0 + 1
        # print('-----',box_h,box_w, y1,y0)

        if min(box_h, box_w) <= obj_size:
            parallel_angle_params = [0]
        else:
            parallel_angle_params = list(range(angle_params[0], angle_params[1], angle_params[2]))
        # print("#angles = {}".format(len(parallel_angle_params)))

        for angle in parallel_angle_params:
            pred_parallel = parallel_transform(pred, box_h, box_w, angle, is_mask=False)
            msk0, msk1 = parallel_transform(msk, box_h, box_w, angle, is_mask=True)
            pred_parallel0 = pred_parallel * msk0
            pred_parallel1 = pred_parallel * msk1
            flag0 = torch.sum(msk0[0], dim=0) > 0.5
            prob0 = torch.max(pred_parallel0[0], dim=0)[0]
            prob0 = prob0[flag0]
            flag1 = torch.sum(msk1[0], dim=1) > 0.5
            prob1 = torch.max(pred_parallel1[0], dim=1)[0]
            prob1 = prob1[flag1]
            if len(prob0) > 0:
                ypred_pos[c.item()].append(prob0)
            if len(prob1) > 0:
                ypred_pos[c.item()].append(prob1)
            # print(nb_ob,angle,len(prob0),len(prob1))
            # print(torch.unique(torch.sum(msk0[0], dim=0)))
            # print(torch.unique(torch.sum(msk1[0], dim=1)))
        #     plt.figure()
        #     plt.subplot(1,2,1)
        #     plt.imshow(msk0[0].cpu().numpy())
        #     plt.subplot(1,2,2)
        #     plt.imshow(msk1[0].cpu().numpy())
        #     plt.savefig('mask_'+str(angle)+'.png')
        # import sys
        # sys.exit()

    if mode == 'focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask  # weights for negative samples
        weight = weight * (torch.rand(ypred.shape, dtype=ypred.dtype, device=ypred.device) < sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:, c, :, :]
            y_neg = y_neg[(mask[:, c, :, :] < 0.5) & (weight[:, c, :, :] > 0.5)]
            bce_neg = -(1 - alpha) * (y_neg ** gamma) * torch.log(1 - y_neg)
            if len(ypred_pos[c]) > 0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1 - epsilon)
                bce_pos = -alpha * ((1 - y_pos) ** gamma) * torch.log(y_pos)
                loss = (bce_neg.sum() + bce_pos.sum()) / len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode == 'mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        v1 = torch.max(ypred * (1 - mask), dim=2)[0]
        v2 = torch.max(ypred * (1 - mask), dim=3)[0]
        ypred_neg = torch.cat([v1, v2], dim=-1).permute(1, 0, 2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0], -1))

        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1 - alpha) * (ypred_neg ** gamma) * torch.log(1 - ypred_neg)
            if len(ypred_pos[c]) > 0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha * ((1 - y_pos) ** gamma) * torch.log(y_pos)
                loss = (bce_neg.sum() + bce_pos.sum()) / len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        ## for negative class
        v1 = torch.max(ypred * (1 - mask), dim=2)[0]
        v2 = torch.max(ypred * (1 - mask), dim=3)[0]
        ypred_neg = torch.cat([v1, v2], dim=-1).permute(1, 0, 2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0], -1))

        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1 - ypred_neg[c])
            if len(ypred_pos[c]) > 0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1 - epsilon)
                bce_pos = -torch.log(pred)
                if mode == 'all':
                    loss = (bce_pos.sum() + bce_neg.sum()) / (len(bce_pos) + len(bce_neg))
                elif mode == 'balance':
                    loss = (bce_pos.mean() + bce_neg.mean()) / 2
            else:
                loss = bce_neg.mean()
            losses[c] = loss

    return losses


def mil_parallel_approx_sigmoid_loss(ypred, mask, crop_boxes, angle_params=(0, 45, 5), mode='all',
                                     method='gm', gpower=4,
                                     focal_params={'alpha': 0.25, 'gamma': 2.0, 'sampling_prob': 1.0},
                                     obj_size=0, epsilon=1e-6):
    assert (mode == 'all') | (mode == 'balance') | (mode == 'focal') | (mode == 'mil_focal')
    ypred = torch.clamp(ypred, epsilon, 1 - epsilon)
    num_classes = ypred.shape[1]
    ob_img_index = crop_boxes[:, 0].type(torch.int32)
    ob_class_index = crop_boxes[:, 1].type(torch.int32)
    ob_crop_boxes = crop_boxes[:, 2:]
    ypred_pos = {c: [] for c in range(num_classes)}
    for nb_ob in range(crop_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c = ob_class_index[nb_ob]
        radius = ob_crop_boxes[nb_ob, -1]

        extra = 5
        cx, cy, r = ob_crop_boxes[nb_ob, :].type(torch.int32)
        r = r + extra
        xmin = torch.clamp(cx - r, 0)
        ymin = torch.clamp(cy - r, 0)
        pred = ypred[nb_img, c, ymin:cy + r + 1, xmin:cx + r + 1][None, :, :]
        msk = mask[nb_img, c, ymin:cy + r + 1, xmin:cx + r + 1][None, :, :]

        index = torch.nonzero(msk[0] > 0.5, as_tuple=True)
        y0, y1 = index[0].min(), index[0].max()
        x0, x1 = index[1].min(), index[1].max()
        box_h = y1 - y0 + 1
        box_w = x1 - x0 + 1

        if min(box_h, box_w) <= obj_size:
            parallel_angle_params = [0]
        else:
            parallel_angle_params = list(range(angle_params[0], angle_params[1], angle_params[2]))
        # print("parallel_angle_params: ", parallel_angle_params)

        for angle in parallel_angle_params:
            pred_parallel = parallel_transform(pred, box_h, box_w, angle, is_mask=False)
            msk0, msk1 = parallel_transform(msk, box_h, box_w, angle, is_mask=True)
            # plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(msk0[0].cpu().numpy())
            # plt.subplot(1,2,2)
            # plt.imshow(msk1[0].cpu().numpy())
            # plt.savefig('mask_'+str(angle)+'.png')
            pred_parallel = pred_parallel[0]
            msk0 = msk0[0] > 0.5
            msk1 = msk1[0] > 0.5
            flag0 = torch.sum(msk0, dim=0) > 0.5
            flag1 = torch.sum(msk1, dim=1) > 0.5
            pred_parallel0 = pred_parallel[:, flag0]
            pred_parallel1 = pred_parallel[flag1, :]
            msk0 = msk0[:, flag0]
            msk1 = msk1[flag1, :]
            # plt.figure()
            # if torch.sum(flag0)>0.5:
            #     plt.subplot(1,2,1)
            #     plt.imshow(msk0.cpu().numpy())
            # if torch.sum(flag1)>0.5:
            #     plt.subplot(1,2,2)
            #     plt.imshow(msk1.cpu().numpy())
            # plt.savefig('mask_'+str(angle)+'_crop.png')

            if torch.sum(flag0) > 0.5:
                if method == 'gm':
                    w = pred_parallel0 ** gpower
                    prob0 = torch.sum(w * msk0, dim=0) / torch.sum(msk0, dim=0)
                    prob0 = prob0 ** (1.0 / gpower)
                elif method == 'expsumr':
                    w = torch.exp(gpower * pred_parallel0)
                    prob0 = torch.sum(pred_parallel0 * w * msk0, dim=0) / torch.sum(w * msk0, dim=0)
                elif method == 'explogs':
                    w = torch.exp(gpower * pred_parallel0)
                    prob0 = torch.log(torch.sum(w * msk0, dim=0)) / gpower - torch.log(torch.sum(msk0, dim=0)) / gpower
                ypred_pos[c.item()].append(prob0)
            if torch.sum(flag1) > 0.5:
                if method == 'gm':
                    w = pred_parallel1 ** gpower
                    prob1 = torch.sum(w * msk1, dim=1) / torch.sum(msk1, dim=1)
                    prob1 = prob1 ** (1.0 / gpower)
                elif method == 'expsumr':
                    w = torch.exp(gpower * pred_parallel1)
                    prob1 = torch.sum(pred_parallel1 * w * msk1, dim=1) / torch.sum(w * msk1, dim=1)
                elif method == 'explogs':
                    w = torch.exp(gpower * pred_parallel1)
                    prob1 = torch.log(torch.sum(w * msk1, dim=1)) / gpower - torch.log(torch.sum(msk1, dim=1)) / gpower
                ypred_pos[c.item()].append(prob1)
            # print(nb_ob,angle,len(prob0),len(prob1))
        # import sys
        # sys.exit()

    if mode == 'focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask  # weights for negative samples
        weight = weight * (torch.rand(ypred.shape, dtype=ypred.dtype, device=ypred.device) < sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:, c, :, :]
            y_neg = y_neg[(mask[:, c, :, :] < 0.5) & (weight[:, c, :, :] > 0.5)]
            bce_neg = -(1 - alpha) * (y_neg ** gamma) * torch.log(1 - y_neg)
            if len(ypred_pos[c]) > 0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1 - epsilon)
                bce_pos = -alpha * ((1 - y_pos) ** gamma) * torch.log(y_pos)
                loss = (bce_neg.sum() + bce_pos.sum()) / len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode == 'mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        if method == 'gm':
            ypred_g = ypred ** gpower
        elif method == 'expsumr':  # alpha-softmax function
            ypred_g = torch.exp(gpower * ypred)
        elif method == 'explogs':  # alpha-quasimax function
            ypred_g = torch.exp(gpower * ypred)
        ## for negative class
        if method == 'gm':
            v1 = (torch.sum(ypred_g * (1 - mask), dim=2) / torch.sum(1 - mask, dim=2)) ** (1.0 / gpower)
            v2 = (torch.sum(ypred_g * (1 - mask), dim=3) / torch.sum(1 - mask, dim=3)) ** (1.0 / gpower)
        elif method == 'expsumr':
            v1 = torch.sum(ypred * ypred_g * (1 - mask), dim=2) / torch.sum(ypred_g * (1 - mask), dim=2)
            v2 = torch.sum(ypred * ypred_g * (1 - mask), dim=3) / torch.sum(ypred_g * (1 - mask), dim=3)
        elif method == 'explogs':
            v1 = torch.log(torch.sum(ypred_g * (1 - mask), dim=2)) / gpower - torch.log(
                torch.sum(1 - mask, dim=2)) / gpower
            v2 = torch.log(torch.sum(ypred_g * (1 - mask), dim=3)) / gpower - torch.log(
                torch.sum(1 - mask, dim=3)) / gpower
        ypred_neg = torch.cat([v1, v2], dim=-1).permute(1, 0, 2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0], -1))

        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1 - alpha) * (ypred_neg ** gamma) * torch.log(1 - ypred_neg)
            if len(ypred_pos[c]) > 0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha * ((1 - y_pos) ** gamma) * torch.log(y_pos)
                loss = (bce_neg.sum() + bce_pos.sum()) / len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        if method == 'gm':
            ypred_g = ypred ** gpower
        elif method == 'expsumr':  # alpha-softmax function
            ypred_g = torch.exp(gpower * ypred)
        elif method == 'explogs':  # alpha-quasimax function
            ypred_g = torch.exp(gpower * ypred)
        ## for negative class
        if method == 'gm':
            v1 = (torch.sum(ypred_g * (1 - mask), dim=2) / torch.sum(1 - mask, dim=2)) ** (1.0 / gpower)
            v2 = (torch.sum(ypred_g * (1 - mask), dim=3) / torch.sum(1 - mask, dim=3)) ** (1.0 / gpower)
        elif method == 'expsumr':
            v1 = torch.sum(ypred * ypred_g * (1 - mask), dim=2) / torch.sum(ypred_g * (1 - mask), dim=2)
            v2 = torch.sum(ypred * ypred_g * (1 - mask), dim=3) / torch.sum(ypred_g * (1 - mask), dim=3)
        elif method == 'explogs':
            v1 = torch.log(torch.sum(ypred_g * (1 - mask), dim=2)) / gpower - torch.log(
                torch.sum(1 - mask, dim=2)) / gpower
            v2 = torch.log(torch.sum(ypred_g * (1 - mask), dim=3)) / gpower - torch.log(
                torch.sum(1 - mask, dim=3)) / gpower
        ypred_neg = torch.cat([v1, v2], dim=-1).permute(1, 0, 2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0], -1))

        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1 - ypred_neg[c])
            if len(ypred_pos[c]) > 0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1 - epsilon)
                bce_pos = -torch.log(pred)
                if mode == 'all':
                    loss = (bce_pos.sum() + bce_neg.sum()) / (len(bce_pos) + len(bce_neg))
                elif mode == 'balance':
                    loss = (bce_pos.mean() + bce_neg.mean()) / 2
            else:
                loss = bce_neg.mean()
            losses[c] = loss

    return losses


def mil_pairwise_loss(ypred, mask, softmax=True, exp_coef=-1):
    """ Compute the pair-wise loss.

        As defined in Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior

    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
    Returns
        pair-wise loss for each category (C,)
    """
    device = ypred.device
    center_weight = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    pairwise_weights_list = [
        torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]),
        torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]]),
        torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
        torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 1., 0.]]),
        torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
        torch.tensor([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]]),
        torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]),
        torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]])]
    ## pairwise loss for each col/row MIL
    num_classes = ypred.shape[1]
    if softmax:
        num_classes = num_classes - 1
    losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=device)
    for c in range(num_classes):
        pairwise_loss = []
        for w in pairwise_weights_list:
            weights = center_weight - w
            weights = weights.view(1, 1, 3, 3).to(device)
            aff_map = F.conv2d(ypred[:, c, :, :].unsqueeze(1), weights, padding=1)
            cur_loss = aff_map ** 2
            if exp_coef > 0:
                cur_loss = torch.exp(exp_coef * cur_loss) - 1
            cur_loss = torch.sum(cur_loss * mask[:, c, :, :].unsqueeze(1)) / (torch.sum(mask[:, c, :, :] + 1e-6))
            pairwise_loss.append(cur_loss)
        losses[c] = torch.mean(torch.stack(pairwise_loss))
    return losses


torch.pi = torch.acos(torch.zeros(1))[0] * 2


def _get_inverse_affine_matrix(
        center: List[float], angle: float, translate: List[float], scale: float, shear: List[float]
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix


def rotate(
        img: Tensor, angle: float, resample: int = 0, expand: bool = False,
        center: Optional[List[int]] = None, fill: Optional[int] = None
) -> Tensor:
    """Rotate the image by angle.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): image to be rotated.
        angle (float or int): rotation angle value in degrees, counter-clockwise.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (list or tuple, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.
            This option is not supported for Tensor input. Fill value for the area outside the transform in the output
            image is always 0.

    Returns:
        PIL Image or Tensor: Rotated image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """
    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    center_f = [0.0, 0.0]
    if center is not None:
        img_size = functional_tensor._get_image_size(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, img_size)]

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
    return functional_tensor.rotate(img, matrix=matrix, resample=resample, expand=expand, fill=fill)


def parallel_transform(image, box_height, box_width, angle, is_mask=True, epsilon=1e-6):
    if abs(angle) > epsilon:
        image_rot = rotate(image, angle, resample=2, expand=True)
    else:
        image_rot = image.clone()

    if is_mask:
        scale = 1 / torch.cos(angle / 180. * torch.pi)
        rot_h = torch.floor(box_height * scale)
        rot_w = torch.floor(box_width * scale)
        # print('**********',angle,scale,rot_h,rot_w)
        # print(torch.sum(image_rot>=0.5,dim=(0,1)))
        # print(torch.sum(image_rot>=0.5,dim=(0,2)))

        flag = torch.sum(image_rot > 0.5, dim=(0, 1)) < rot_h - 0.5
        rot0 = image_rot.clone()
        rot0[:, :, flag] = 0

        flag = torch.sum(image_rot > 0.5, dim=(0, 2)) < rot_w - 0.5
        rot1 = image_rot.clone()
        rot1[:, flag, :] = 0
        return rot0, rot1
    else:
        return image_rot