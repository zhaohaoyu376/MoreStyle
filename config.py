import torch
import numpy as np
import torch.nn as nn


def dice_loss(pred, label):
    smooth = 1.
    bs = pred.size(0)
    m1 = pred.contiguous().view(bs, -1)
    m2 = label.contiguous().view(bs, -1)
    intersection = (m1 * m2).sum()
    loss = 1 - ((2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth))
    return loss

def bce_loss(pred, label):
    score = torch.nn.BCEWithLogitsLoss()(pred, label)
    return score

class Seg_loss(nn.Module):
    def __init__(self):
        super(Seg_loss, self).__init__()

    def forward(self, logit_pred, label):
        pred = torch.nn.Sigmoid()(logit_pred)
        score = dice_loss(pred=pred, label=label) + bce_loss(pred=pred, label=label)
        return score


def weighted_dice_loss(pred, label, uncertainty):
    smooth = 1.
    bs = pred.size(0)
    m1 = pred.contiguous().view(bs, -1)
    m2 = label.contiguous().view(bs, -1)

    intersection = (m1 * m2).sum()
    dice_coeff = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    weighted_dice = dice_coeff * (1-uncertainty)

    loss = 1.0 - weighted_dice
    return loss.mean()

def weighted_bce_loss(pred, label, uncertainty):
    # bce_loss = torch.nn.BCEWithLogitsLoss()(pred, label)
    # weighted_loss = (uncertainty * bce_loss).mean()
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, label, reduction='none')
    weighted_loss = bce_loss * uncertainty
    return weighted_loss.mean()

class weighted_Seg_loss(nn.Module):
    def __init__(self):
        super(weighted_Seg_loss, self).__init__()

    def forward(self, logit_pred, label,uncert):
        pred = torch.nn.Sigmoid()(logit_pred)
        # score = weighted_dice_loss(pred=pred, label=label,uncertainty=uncert) + weighted_bce_loss(pred=pred, label=label,uncertainty=uncert)
        score = dice_loss(pred=pred, label=label) + weighted_bce_loss(pred=pred, label=label,uncertainty=uncert)
        return score


class EpochLR(torch.optim.lr_scheduler._LRScheduler):
    # lr_n = lr_0 * (1 - epoch / epoch_nums)^gamma
    def __init__(self, optimizer, epochs, gamma=0.9, last_epoch=-1):
        self.lr = optimizer.param_groups[0]['lr']
        self.epochs = epochs
        self.gamma = gamma
        super(EpochLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lr * pow((1. - (self.last_epoch + 1) / self.epochs), self.gamma)]

