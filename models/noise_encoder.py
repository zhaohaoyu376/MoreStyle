import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import sys
import random
sys.path.append('../../')


class _DomainSpecificBatchNorm(nn.Module):
    """
    code is based on https://github.com/wgchang/DSBN/blob/e0cd4bf48f9a6f2a2f4f31e6e88e00abc14049c0/model/resnetdsbn.py#L225
    """
    _version = 2

    def __init__(self, num_features, num_domains, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()
        #         self.bns = nn.ModuleList([nn.modules.batchnorm._BatchNorm(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features, eps, momentum, affine,
                                                 track_running_stats) for _ in range(num_domains)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_id):
        self._check_input_dim(x)
        x = self.bns[domain_id](x)
        return x

class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class Fixable2DDropout(nn.Module):
    """
    _summary_method = torch.nn.Dropout2d.__init__
     based on 2D pytorch mask, supports lazy load with last generated mask
    """
    def __init__(self, p: float = 0.5,inplace=False,lazy_load: bool = False,training=True):
        super(Fixable2DDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.seed  = None
        self.lazy_load = lazy_load
        self.training=training

    def forward(self, X):
        if self.training:
            if self.lazy_load:
                if not self.seed is None:
                    seed  = self.seed
                else:
                    seed = torch.seed()
            else:seed = torch.seed()
        else:
            seed = torch.seed()
        self.seed=seed
        torch.manual_seed(seed)
        X = F.dropout2d(X, p=self.p, training=self.training, inplace=self.inplace)
        return X

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class res_convdown(nn.Module):
    def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None):
        super(res_convdown, self).__init__()
        # down-> conv3->prelu->conv
        if if_SN:
            self.down = spectral_norm(nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, bias=bias))

            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias)),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias)),
                norm(out_ch),
            )
        else:
            self.down = (nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, bias=bias))

            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
                # nn.LeakyReLU(0.2),
                # nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
                # norm(out_ch),
            )
        if if_SN:
            self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias))
        else:
            self.conv_input = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

        self.last_act = nn.LeakyReLU(0.2)
        self.dropout = dropout
        if not self.dropout is None:
            # self.drop = nn.Dropout2d(p=dropout)
            self.drop = Fixable2DDropout(p=dropout)

    def get_features(self, x):
        x = self.down(x)
        res_x = self.conv_input(x) + self.conv(x)
        return res_x

    def non_linear(self, x):
        res_x = self.last_act(x)
        if not self.dropout is None:
            res_x = self.drop(res_x)
        return res_x

    def forward(self, x):
        x = self.get_features(x)
        x = self.non_linear(x)
        return x

class res_NN_up(nn.Module):
    '''
    upscale with NN upsampling followed by conv
    '''

    def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None):
        super(res_NN_up, self).__init__()
        # up-> conv3->prelu->conv
        self.up = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=bias)
        )

        if if_SN:

            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias), dim=1),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias), dim=1),
                norm(out_ch),
            )
        else:

            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
            )

        if if_SN:
            self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                                      stride=1, padding=0, bias=bias), dim=1)
        else:
            self.conv_input = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

        self.last_act = nn.LeakyReLU(0.2)
        self.dropout = dropout
        if not self.dropout is None:
            self.drop= Fixable2DDropout(p=dropout)

            # self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.up(x)
        res_x = self.last_act(self.conv_input(x) + self.conv(x))
        if not self.dropout is None:
            res_x = self.drop(res_x)
        # appl
        return res_x

class res_up_family(nn.Module):
    '''
    upscale with different upsampling methods
    '''

    def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, if_SN=False, bias=True, dropout=None, up_type='bilinear'):
        super(res_up_family, self).__init__()
        # up-> conv3->prelu->conv
        if up_type == 'NN':
            self.up = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
            )
        elif up_type == 'bilinear':
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
            )
        elif up_type == 'Conv2':
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        elif up_type == 'Conv4':
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)
        else:
            raise NotImplementedError

        if if_SN:

            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias), dim=1),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias), dim=1),
                norm(out_ch),
            )
        else:

            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=bias),
                norm(out_ch),
            )

        if if_SN:
            self.conv_input = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                                      stride=1, padding=0, bias=bias), dim=1)
        else:
            self.conv_input = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

        self.last_act = nn.LeakyReLU(0.2)
        self.dropout = dropout
        if not self.dropout is None:
            self.drop= Fixable2DDropout(p=dropout)
            # self.drop = nn.Dropout2d(p=dropout)

    def get_features(self, x):
        x = self.up(x)
        res_x = self.conv_input(x) + self.conv(x)
        return res_x

    def non_linear(self, x):
        res_x = self.last_act(x)
        if not self.dropout is None:
            res_x = self.drop(res_x)
        return res_x

    def forward(self, x):
        x = self.get_features(x)
        x = self.non_linear(x)
        return x

class ds_res_convdown(nn.Module):
    '''
    res conv down with domain specific layers
    '''

    def __init__(self, in_ch, out_ch, num_domains=2, if_SN=False, bias=True, dropout=None):
        super(ds_res_convdown, self).__init__()
        # down-> conv3->prelu->conv
        if if_SN:
            self.down = spectral_norm(
                nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, bias=bias))

            self.conv_1 = spectral_norm(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias))
            self.norm_1 = DomainSpecificBatchNorm2d(
                out_ch, num_domains=num_domains)
            self.act_1 = nn.LeakyReLU(0.2)
            self.conv_2 = spectral_norm(nn.Conv2d(out_ch, out_ch,
                                                  3, padding=1, bias=bias))

            self.norm_2 = DomainSpecificBatchNorm2d(
                out_ch, num_domains=num_domains)
        else:
            self.down = nn.Conv2d(
                in_ch, in_ch, 3, stride=2, padding=1, bias=bias)

            self.conv_1 = spectral_norm(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=bias))
            self.norm_1 = DomainSpecificBatchNorm2d(
                out_ch, num_domains=num_domains)
            self.act_1 = nn.LeakyReLU(0.2)
            self.conv_2 = nn.Conv2d(out_ch, out_ch,
                                    3, padding=1, bias=bias)

            self.norm_2 = DomainSpecificBatchNorm2d(
                out_ch, num_domains=num_domains)

        if if_SN:
            self.conv_input = spectral_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias))
        else:
            self.conv_input = nn.Conv2d(
                in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)

        self.last_act = nn.LeakyReLU(0.2)
        self.dropout = dropout
        if not self.dropout is None:
            self.drop= Fixable2DDropout(p=dropout)
            # self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x, domain_id=0):
        x = self.down(x)
        f = self.conv_1(x)
        f = self.norm_1(f, domain_id)
        f = self.act_1(f)
        f = self.conv_2(f)
        f = self.norm_2(f, domain_id)
        res_x = self.last_act(self.conv_input(x) + f)
        if not self.dropout is None:
            res_x = self.drop(res_x)
        return res_x

class NoiseEncoder(nn.Module):
    def __init__(self, input_channel, output_channel=None, feature_reduce=2, encoder_dropout=None, norm=nn.InstanceNorm2d, if_SN=False, act=torch.nn.Sigmoid()):
        super(NoiseEncoder, self).__init__()
        if if_SN:
            self.inc = nn.Sequential(
                spectral_norm(nn.Conv2d(input_channel, 64 // feature_reduce, 3, padding=1, bias=True)),
                norm(64 // feature_reduce),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64 // feature_reduce, 64 // feature_reduce, 3, padding=1, bias=True),
                norm(64 // feature_reduce),
            )
        else:
            self.inc = nn.Sequential(
                nn.Conv2d(input_channel, 64 // feature_reduce, 3, padding=1, bias=True),
                norm(64 // feature_reduce),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64 // feature_reduce, 64 // feature_reduce, 3, padding=1, bias=True),
                norm(64 // feature_reduce),
            )

        self.fft_noise = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, padding=1, bias=True),
            norm(64 // feature_reduce),
        )

        self.down1 = res_convdown(64 // feature_reduce, 128 // feature_reduce,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down2 = res_convdown(128 // feature_reduce, 128 // feature_reduce,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down3 = res_convdown(128 // feature_reduce, 256 // feature_reduce,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down4 = res_convdown(256 // feature_reduce, 512 // feature_reduce,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        self.down5 = res_convdown(512 // feature_reduce, 1024 // feature_reduce,
                                  norm=norm, if_SN=if_SN, dropout=encoder_dropout)
        if output_channel is None:
            self.final_conv = nn.Sequential(
                nn.Conv2d(512 // feature_reduce, 512 // feature_reduce, kernel_size=1, stride=1, padding=0),
                norm(512 // feature_reduce))
        else:
            self.final_conv = nn.Sequential(
                nn.Conv2d(512 // feature_reduce, output_channel, kernel_size=1, stride=1, padding=0),
                norm(output_channel))

        self.act = act
        for m in self._modules:
            normal_init(self._modules[m], 0, 0.02)

        self.y2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
                norm(output_channel))
        self.y3 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
                norm(output_channel))
        self.y4 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
                norm(output_channel))
        self.y5 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                norm(output_channel))
        self.y6 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                norm(output_channel))

    def forward(self, x, domain_id=0):
        x1 = self.inc(x)
        x1 = F.leaky_relu(x1, negative_slope=0.2)
        x2 = self.down1(x1)
        y2 = self.y2(x2)
        x3 = self.down2(x2)
        y3 = self.y3(x3)
        x4 = self.down3(x3)
        y4 = self.y4(x4)
        x5 = self.down4(x4)
        y5 = self.y5(x5)
        x6 = self.down5(x5)
        y6 = self.y6(x6)

        return [x2,x3,x4,x5,x6,y2,y3,y4,y5,y6]


def extract_amp_spectrum(img_np):
    # trg_img is of dimention CxHxW (C = 3 for RGB image and 1 for slice)
    fft = np.fft.fft2(img_np, axes=(-2, -1))
    amp_np, pha_np = np.abs(fft), np.angle(fft)
    return amp_np

def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    ratio = random.randint(1, 20) / 100
    # ratio = 0.2
    # ratio = 1
    a_trg = torch.randn_like(torch.tensor(a_trg))
    a_trg = a_trg.numpy()

    a_src[:, h1:h2, w1:w2] = a_src[:, h1:h2, w1:w2] * (1 - ratio) + a_trg[:, h1:h2, w1:w2] * ratio
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src

def source_to_target_freq(src_img, amp_trg, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img
    fft_src_np = np.fft.fft2(src_img, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_src = np.abs(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg

def FFT_noise(img_clear,noise):
    img_res = img_clear[0]
    other_img = noise[0]
    amp_trg = extract_amp_spectrum(other_img.astype(np.float32))

    img_freq = source_to_target_freq(img_res, amp_trg, L=0.1)
    img_freq = np.clip(img_freq, 0, 255).astype(np.float32)

    img_freq = torch.from_numpy(img_freq).float()
    img_res = torch.unsqueeze(img_freq, dim=0)

    if(len(img_clear)>1):
        for i in range(1, len(img_clear)):
            img = img_clear[0]
            other_img = noise[0]
            amp_trg = extract_amp_spectrum(other_img.astype(np.float32))

            img_freq = source_to_target_freq(img, amp_trg, L=0.1)
            img_freq = np.clip(img_freq, 0, 255).astype(np.float32)

            img_freq = torch.from_numpy(img_freq).float()
            img_freq = torch.unsqueeze(img_freq, dim=0)
            img_res = torch.cat([img_res, img_freq], dim=0)

    return img_res

if __name__ == '__main__':
    # encoder = Dual_Branch_Encoder(input_channel=3, z_level_1_channel=128, z_level_2_channel=128,
    #                     feature_reduce=4, if_SN=False, encoder_dropout=None,
    #                     norm=nn.BatchNorm2d, num_domains=1)
    # image = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    # z_i,z_s = encoder(image,1)
    # print(z_i.size())
    general_encoder = NoiseEncoder(3)
    input = torch.randn([8,3,512,512])
    [x2,x3,x4,x5,x6,y2,y3,y4,y5,y6] = general_encoder(input)
    print(x2.size())
    print(x3.size())
    print(x4.size())
    print(x5.size())
    print(x6.size())