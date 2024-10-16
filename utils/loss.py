import torchvision
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import sys
from PIL import Image

def cross_entropy_2D(input, target, weight=None, size_average=True, mask=None, is_gt=False):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    if mask is None:
        mask = torch.ones(n, 1, h, w, device=log_p.device, requires_grad=False)
    mask = mask.reshape(n * h * w, 1)
    mask.requires_grad = False
    mask_region_size = float(torch.numel(mask))
    if not weight is None:
        weight = np.array(weight)
        weight = weight / (1.0 * sum(weight)) * c
        weight = torch.tensor(weight, device=input.device, dtype=torch.float32)
    if len(target.size()) == 3:
        target = target.view(target.numel())
        loss_vector = F.nll_loss(
            log_p, target, weight=weight, reduction="none")
        # print(loss_vector.size())
        loss_vector = loss_vector * mask.flatten()
        loss = torch.sum(loss_vector)
        if size_average:
            loss /= float(mask_region_size)  # /N*H'*W'
    elif len(target.size()) == 4:
        # ce loss=-qlog(p)
        if not is_gt:
            reference = F.softmax(target, dim=1)  # M,C
        else:
            reference = target
        reference = reference.transpose(1, 2).transpose(
            2, 3).contiguous().view(-1, c)  # M,C
        mask = mask.expand(n * h * w, c)
        if weight is None:
            plogq = torch.sum(reference * log_p * mask, dim=1)
            plogq = torch.sum(plogq)
            if size_average:
                plogq /= float(mask_region_size)
        else:
            plogq_class_wise = reference * log_p * mask
            plogq_sum_class = 0.
            for i in range(c):
                plogq_sum_class += torch.sum(plogq_class_wise[:, i] * weight[i])
            plogq = plogq_sum_class
            if size_average:
                # only average loss on the mask entries with value =1
                plogq /= float(mask_region_size)
        loss = -1 * plogq
    else:
        raise NotImplementedError
    return loss

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, inputs, targets, weight_mask):
        loss = torch.mean(weight_mask * (inputs - targets) ** 2)
        return loss

def fourier_loss(input, target, l_w=0.00001, L=0.1):
    input = input.float()
    B,C,h,w = target.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)

    target_fft = torch.fft.fft2(target)
    input_fft = torch.fft.fft2(input)

    amp_input, pha_input = torch.abs(input_fft), torch.angle(input_fft)
    amp_target, pha_target = torch.abs(target_fft), torch.angle(target_fft)

    amp_input = torch.fft.fftshift(amp_input)
    amp_target = torch.fft.fftshift(amp_target)

    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    weight = torch.ones(target.size())
    weight[:,:,h1:h2,w1:w2] = 0.1
    weight = weight.to(input.device)

    # amp_loss = WeightedMSELoss()(input,amp_target,weight)
    # amp_input = amp_target * torch.exp(1j * pha_target)
    # src_in_trg = torch.fft.ifft2(amp_input)
    # src_in_trg = torch.real(src_in_trg)
    # print(src_in_trg.size())
    #
    # print(src_in_trg[0])
    # print(target[0])

    amp_loss = WeightedMSELoss()(amp_input,amp_target,weight)
    pha_loss = nn.MSELoss()(pha_input,pha_target)

    loss = pha_loss + l_w*amp_loss
    return loss

def fourier_exchage_loss(input, target, l_w=1e-8, L=0.1,alpha=0.3,l_l=1e-2):
    L = round(0.05 + np.random.random() * L, 2)

    input = input.float()
    B,C,h,w = target.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)

    target_fft = torch.fft.fft2(target)
    input_fft = torch.fft.fft2(input)

    amp_input, pha_input = torch.abs(input_fft), torch.angle(input_fft)
    amp_target, pha_target = torch.abs(target_fft), torch.angle(target_fft)

    amp_input = torch.fft.fftshift(amp_input)
    amp_target = torch.fft.fftshift(amp_target)

    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    # caculate loss
    weight = torch.ones(target.size())
    weight[:,:,h1:h2,w1:w2] = l_l
    weight = weight.to(input.device)
    amp_loss = WeightedMSELoss()(amp_input,amp_target,weight)
    pha_loss = nn.MSELoss()(pha_input,pha_target)

    # # exchanged img
    # amp_input[:, :, h1:h2, w1:w2] = amp_target[:, :, h1:h2, w1:w2]
    # amp_input = torch.fft.ifftshift(amp_input)
    # amp_target = torch.fft.ifftshift(amp_target)
    # amp_input = amp_input * torch.exp(1j * pha_input)
    # exchanged_img = torch.fft.ifft2(amp_input)
    # exchanged_img = torch.real(exchanged_img)

    # exchanged img
    amp_exchange = alpha*amp_input + (1-alpha)*amp_target
    amp_exchange[:, :, h1:h2, w1:w2] = amp_input[:, :, h1:h2, w1:w2]
    amp_exchange = torch.fft.ifftshift(amp_exchange)
    pha_exchange = alpha*pha_input + (1-alpha)*pha_target
    amp_exchange = amp_exchange * torch.exp(1j * pha_exchange)
    exchanged_img = torch.fft.ifft2(amp_exchange)
    exchanged_img = torch.real(exchanged_img)

    loss = pha_loss + l_w*amp_loss
    return loss, exchanged_img

class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        
        pred = torch.sigmoid(pred)

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        #pred = torch.sigmoid(pred)
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss

if __name__ == '__main__':
    img = Image.open('E:\code\CCSDG-master\CCSDG-master\ccsdg\RIGAPlus\RIGA\BinRushed\BinRushed1/image1.tif')
    img = np.array(img).transpose(2, 0, 1).astype(np.float32)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img,dim=0)
    print(img.size())
    a = torch.randn([1,3,800,800])
    b = torch.randn([1,3,800,800])
    loss = fourier_loss(a,img)
    print(loss)

