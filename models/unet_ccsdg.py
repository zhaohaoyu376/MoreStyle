# -*- coding:utf-8 -*-
from models.unet import UnetBlock, SaveFeatures
from models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
from torch import nn
import torch
import torch.nn.functional as F
from utils.utils import _disable_tracking_bn_stats


class Projector(nn.Module):
    def __init__(self, output_size=1024):
        super(Projector, self).__init__()
        self.conv = nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(131072, output_size)

    def forward(self, x_in):
        x = self.conv(x_in)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

class UNetCCSDG(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=1, pretrained=False, ini=1):
        super().__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')
        self.base_model = base_model(ini=ini)
        layers = list(base_model(pretrained=pretrained).children())[:cut]
        first_layer = layers[0]
        other_layers = layers[1:]
        base_layers = nn.Sequential(*other_layers)
        self.first_layer = first_layer
        self.rn = base_layers

        self.channel_prompt = nn.Parameter(torch.randn(2, 64, 1, 1))

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [1, 3, 4, 5]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

    def forward_first_layer(self, x, tau=0.1):
        x = self.first_layer(x)  # 8 64 256 256

        channel_prompt_onehot = torch.softmax(self.channel_prompt/tau, dim=0)
        f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)
        return f_content, f_style

    def encoder(self,x,tau=0.1):
        x = self.first_layer(x)
        channel_prompt_onehot = torch.softmax(self.channel_prompt / tau, dim=0)
        f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)
        x = self.rn(f_content)
        sfs = [self.sfs[i].features for i in [0,1,2,3]]
        return x, sfs

    def forward(self, x, tau=0.1):
        x = self.first_layer(x)  # 8 64 256 256

        channel_prompt_onehot = torch.softmax(self.channel_prompt / tau, dim=0)
        f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)
        x = F.relu(self.rn(f_content))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        output = self.up5(x)

        return output

    def close(self):
        for sf in self.sfs: sf.remove()

class UNetTriD(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=1, pretrained=False, ini=3,
                 mixstyle_layers=[],random_type=None,p=0.5):
        super().__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        self.res = base_model(ini=ini,mixstyle_layers=mixstyle_layers,random_type=random_type,p=p)

        self.channel_prompt = nn.Parameter(torch.randn(2, 64, 1, 1))

        self.num_classes = num_classes
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)


    # def forward_first_layer(self, x, tau=0.1):
    #     x = self.first_layer(x)  # 8 64 256 256
    #
    #     channel_prompt_onehot = torch.softmax(self.channel_prompt/tau, dim=0)
    #     f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
    #     f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)
    #     return f_content, f_style

    def encoder(self,input):
        x, sfs = self.res(input,True)
        return x, sfs

    def forward(self, input):
        x, sfs = self.res(input)
        x = F.relu(x)

        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        x = self.up5(x)

        return x

    def close(self):
        for sf in self.sfs:
            sf.remove()

class UNetMaxStyle(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=1, pretrained=False, ini=3,
                 mixstyle_layers=[],random_type=None,p=0.5):
        super().__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        self.res = base_model(ini=ini,mixstyle_layers=mixstyle_layers,random_type=random_type,p=p)

        self.channel_prompt = nn.Parameter(torch.randn(2, 64, 1, 1))

        self.num_classes = num_classes
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

        # generater
        self.up1_generator = UnetBlock(512, 256, 256)
        self.up2_generator = UnetBlock(256, 128, 256)
        self.up3_generator = UnetBlock(256, 64, 256)
        self.up4_generator = UnetBlock(256, 64, 256)
        self.generate_img = nn.ConvTranspose2d(256, 3, 2, stride=2)


    def encoder(self,input):
        x, sfs = self.res(input,True)
        return x, sfs

    def forward(self, input):
        x, sfs = self.res(input)
        x = F.relu(x)

        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        x = self.up5(x)

        return x

    def generator(self, x,sfs):
        x = F.relu(x)

        x = self.up1_generator(x, sfs[3])
        x = self.up2_generator(x, sfs[2])
        x = self.up3_generator(x, sfs[1])
        x = self.up4_generator(x, sfs[0])

        output = self.generate_img(x)

        return output

    def apply_max_style(self, x,sfs, nn_style_augmentor_dict, decoder_layers_indexes=[3, 4, 5]):
        if 0 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(0)](x.detach().clone())
        else:
            x = x.detach().clone()

        x = F.relu(x)

        with _disable_tracking_bn_stats(self.up1_generator):
            x = self.up1_generator(x, sfs[3])
        if 1 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(1)](x)

        with _disable_tracking_bn_stats(self.up2_generator):
            x = self.up2_generator(x, sfs[2])
        if 2 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(2)](x)

        with _disable_tracking_bn_stats(self.up3_generator):
            x = self.up3_generator(x, sfs[1])

        if 3 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(3)](x)

        with _disable_tracking_bn_stats(self.up4_generator):
            x = self.up4_generator(x, sfs[0])
        if 4 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(4)](x)

        with _disable_tracking_bn_stats(self.generate_img):
            x = self.generate_img(x)
        if 5 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(5)](x)

        return x


    def close(self):
        for sf in self.sfs:
            sf.remove()


class UNetFreeSeg(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=1, pretrained=False, ini=3,):
        super().__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')
        self.res = base_model(ini=ini)

        self.channel_prompt = nn.Parameter(torch.randn(2, 64, 1, 1))

        self.num_classes = num_classes
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 3, 2, stride=2)

        self.up11 = UnetBlock(512, 256, 256)
        self.up22 = UnetBlock(256, 128, 256)
        self.up33 = UnetBlock(256, 64, 256)
        self.up44 = UnetBlock(256, 64, 256)
        self.up55 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

    def encoder(self,input):
        x, sfs = self.res(input,True)
        return x, sfs

    def forward(self, input):
        x, sfs = self.res(input)
        x1 = F.relu(x)

        x = self.up1(x1, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        x = self.up5(x)

        xx = self.up11(x1, sfs[3])
        xx = self.up22(xx, sfs[2])
        xx = self.up33(xx, sfs[1])
        xx = self.up44(xx, sfs[0])
        xx = self.up55(xx)

        return x,xx

    def close(self):
        for sf in self.sfs:
            sf.remove()


if __name__ == '__main__':
    ccsdg = UNetCCSDG()
    x = torch.randn([8,3,512,512])
    out = ccsdg(x)
    print('out ',out.size())