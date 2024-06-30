from torch import nn
import torch
import torch.nn.functional as F
# from models.unet import UnetBlock, SaveFeatures
# from models.resnet import resnet34_mix, resnet18, resnet50, resnet101, resnet152
from models.unet import UnetBlock, SaveFeatures
from models.resnet import resnet34_mix, resnet18, resnet50, resnet101, resnet152
from models.mix import *
from models.noise_encoder import NoiseEncoder
from models.model_utils import _disable_tracking_bn_stats

class UNet_Mix(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False, mixstyle_layers=[],random_type='MixNoise',ini=3):
        super().__init__()
        self.mixstyle_layers = mixstyle_layers
        # self.noiseEncoder = NoiseEncoder(input_channel=ini)

        if resnet == 'resnet34':
            base_model = resnet34_mix
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

        self.res = base_model(pretrained=pretrained, mixstyle_layers=[], random_type=random_type, p=0.5)

        self.channel_prompt = nn.Parameter(torch.randn(2, 64, 1, 1))

        self.num_classes = num_classes
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        self.generate_img = nn.ConvTranspose2d(256, 3, 2, stride=2)

        if mixstyle_layers:
            if random_type == 'MixNoise':
                self.random = MixNoise(p=0.5)
            elif random_type == 'TriD':
                self.random = TriD(p=0.5)
            elif random_type == 'MixStyle':
                self.random = MixStyle(p=0.5, mix='random')
            elif random_type == 'EFDMixStyle':
                self.random = EFDMix(p=0.5, mix='random')
            else:
                raise ValueError('The random method type is wrong!')
            print('Insert Random Style after the following layers: {}'.format(mixstyle_layers))

    def generate(self,x, isTrain=True):
        noise = torch.randn_like(x)
        # [gamma_noise1,gamma_noise2,gamma_noise3,gamma_noise4,gamma_noise5,
        #  beta_noise1, beta_noise2, beta_noise3, beta_noise4, beta_noise5] = self.noiseEncoder(noise)

        x, sfs = self.res(x, isTrain=True)
        x = F.relu(x)

        # if 'layer0' in self.mixstyle_layers and isTrain:
        #     sfs[3] = self.random(sfs[3],gamma_noise4,beta_noise4)
        x = self.up1(x, sfs[3])

        # if 'layer1' in self.mixstyle_layers and isTrain:
        #     sfs[2] = self.random(sfs[2],gamma_noise3,beta_noise3)
        x = self.up2(x, sfs[2])

        # if 'layer2' in self.mixstyle_layers and isTrain:
        #     sfs[1] = self.random(sfs[1],gamma_noise2,beta_noise2)
        x = self.up3(x, sfs[1])

        # if 'layer3' in self.mixstyle_layers and isTrain:
        #     sfs[0] = self.random(sfs[0],gamma_noise1,beta_noise1)

        x = self.up4(x, sfs[0])
        output = self.generate_img(x)
        return output

    def encoder(self,x, isTrain=True):
        x, sfs = self.res(x, isTrain=True)

        return x,sfs

    def forward(self, x, isTrain=True):
        x, sfs = self.res(x, isTrain=True)

        channel_prompt_onehot = torch.softmax(self.channel_prompt/0.1, dim=0)
        f_content = sfs[0] * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        # f_style = sfs[0] * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)

        # x = self.first_layer(x)
        # channel_prompt_onehot = torch.softmax(self.channel_prompt / tau, dim=0)
        # f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)

        x = F.relu(x)
        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        output = self.up5(x)

        return output, f_content

    def close(self):
        for sf in self.sfs: sf.remove()

class generater(nn.Module):
    def __init__(self, num_classes=2, mixstyle_layers=[],random_type='MixNoise',ini=3):
        super().__init__()
        self.mixstyle_layers = mixstyle_layers
        self.noiseEncoder = NoiseEncoder(input_channel=ini)

        self.channel_prompt = nn.Parameter(torch.randn(2, 64, 1, 1))

        self.num_classes = num_classes
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        self.generate_img = nn.ConvTranspose2d(256, 3, 2, stride=2)

        if mixstyle_layers:
            if random_type == 'MixNoise':
                self.random = MixNoise(p=0.5)
            elif random_type == 'TriD':
                self.random = TriD(p=0.5)
            elif random_type == 'MixStyle':
                self.random = MixStyle(p=0.5, mix='random')
            elif random_type == 'EFDMixStyle':
                self.random = EFDMix(p=0.5, mix='random')
            else:
                raise ValueError('The random method type is wrong!')
            print('Insert Random Style after the following layers: {}'.format(mixstyle_layers))


    def forward(self, x,sfs, data,isTrain=True):
        noise = torch.randn_like(data)
        [gamma_noise1,gamma_noise2,gamma_noise3,gamma_noise4,gamma_noise5,
         beta_noise1, beta_noise2, beta_noise3, beta_noise4, beta_noise5] = self.noiseEncoder(noise)

        # x, sfs = self.res(x, isTrain=True)
        x = F.relu(x)

        if 'layer0' in self.mixstyle_layers and isTrain:
            sfs[3] = self.random(sfs[3],gamma_noise4,beta_noise4)
        x = self.up1(x, sfs[3])

        if 'layer1' in self.mixstyle_layers and isTrain:
            sfs[2] = self.random(sfs[2],gamma_noise3,beta_noise3)
        x = self.up2(x, sfs[2])

        if 'layer2' in self.mixstyle_layers and isTrain:
            sfs[1] = self.random(sfs[1],gamma_noise2,beta_noise2)
        x = self.up3(x, sfs[1])

        if 'layer3' in self.mixstyle_layers and isTrain:
            sfs[0] = self.random(sfs[0],gamma_noise1,beta_noise1)

        x = self.up4(x, sfs[0])
        output = self.generate_img(x)
        return output

class Maxstyle(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False, mixstyle_layers=[],random_type='MixNoise',ini=3):
        super().__init__()
        self.mixstyle_layers = mixstyle_layers
        # self.noiseEncoder = NoiseEncoder(input_channel=ini)

        if resnet == 'resnet34':
            base_model = resnet34_mix
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

        self.res = base_model(pretrained=pretrained, mixstyle_layers=[], random_type=random_type, p=0.5)

        self.channel_prompt = nn.Parameter(torch.randn(2, 64, 1, 1))

        self.num_classes = num_classes
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up11 = UnetBlock(512, 256, 256)
        self.up22 = UnetBlock(256, 128, 256)
        self.up33 = UnetBlock(256, 64, 256)
        self.up44 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        self.generate_img = nn.ConvTranspose2d(256, 3, 2, stride=2)

        if mixstyle_layers:
            if random_type == 'MixNoise':
                self.random = MixNoise(p=0.5)
            elif random_type == 'MixNoise_pro':
                self.random = MixNoise_pro(p=0.5)
            elif random_type == 'TriD':
                self.random = TriD(p=0.5)
            elif random_type == 'MixStyle':
                self.random = MixStyle(p=0.5, mix='random')
            elif random_type == 'EFDMixStyle':
                self.random = EFDMix(p=0.5, mix='random')
            else:
                raise ValueError('The random method type is wrong!')
            print('Insert Random Style after the following layers: {}'.format(mixstyle_layers))

    def encoder(self,x, isTrain=True):
        x, sfs = self.res(x, isTrain=True)

        return x,sfs

    def seg_decoder(self,x,sfs):
        x = F.relu(x)
        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        output = self.up5(x)

        return output

    def img_decoder(self,x,sfs):
        x = F.relu(x)
        x = self.up11(x, sfs[3])
        x = self.up22(x, sfs[2])
        x = self.up33(x, sfs[1])
        x = self.up44(x, sfs[0])
        output = self.generate_img(x)
        return output

    def apply_max_style(self, image_code,sfs, nn_style_augmentor_dict, decoder_layers_indexes=[3, 4]):
        if 0 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(0)](image_code.detach().clone())
        else:
            x = image_code.detach().clone()

        with _disable_tracking_bn_stats(self.up11):
            x2 = self.up11(x,sfs[3])
        if 1 in decoder_layers_indexes:
            x2 = nn_style_augmentor_dict[str(1)](x2)

        with _disable_tracking_bn_stats(self.up22):
            x3 = self.up22(x2,sfs[2])
        if 2 in decoder_layers_indexes:
            x3 = nn_style_augmentor_dict[str(2)](x3)

        with _disable_tracking_bn_stats(self.up33):
            x4 = self.up33(x3,sfs[1])
        if 3 in decoder_layers_indexes:
            x4 = nn_style_augmentor_dict[str(3)](x4)

        with _disable_tracking_bn_stats(self.up44):
            x5 = self.up44(x4,sfs[0])
        if 4 in decoder_layers_indexes:
            x5 = nn_style_augmentor_dict[str(4)](x5)

        with _disable_tracking_bn_stats(self.generate_img):
            x5 = self.generate_img(x5)

        if 5 in decoder_layers_indexes:
            x5 = nn_style_augmentor_dict[str(5)](x5)
        return x5

    def seg_decoder_maxstyle(self, image_code,sfs, nn_style_augmentor_dict, decoder_layers_indexes=[3, 4]):
        if 0 in decoder_layers_indexes:
            x = nn_style_augmentor_dict[str(0)](image_code.detach().clone())
        else:
            x = image_code.detach().clone()
        with _disable_tracking_bn_stats(self.up11):
            x2 = self.up1(x,sfs[3])

        if 1 in decoder_layers_indexes:
            x2 = nn_style_augmentor_dict[str(1)](x2)
        with _disable_tracking_bn_stats(self.up22):
            x3 = self.up2(x2,sfs[2])

        if 2 in decoder_layers_indexes:
            x3 = nn_style_augmentor_dict[str(2)](x3)
        with _disable_tracking_bn_stats(self.up33):
            x4 = self.up3(x3,sfs[1])

        if 3 in decoder_layers_indexes:
            x4 = nn_style_augmentor_dict[str(3)](x4)
        with _disable_tracking_bn_stats(self.up44):
            x5 = self.up4(x4,sfs[0])

        if 4 in decoder_layers_indexes:
            x5 = nn_style_augmentor_dict[str(4)](x5)
        with _disable_tracking_bn_stats(self.generate_img):
            x5 = self.generate_img(x5)

        if 5 in decoder_layers_indexes:
            x5 = nn_style_augmentor_dict[str(5)](x5)
        return x5

    def forward(self, x, isTrain=True):
        x, sfs = self.res(x, isTrain=True)
        img = x.clone()
        x = F.relu(x)
        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        output = self.up5(x)

        return output, img, sfs

    def close(self):
        for sf in self.sfs: sf.remove()

if __name__ == '__main__':
    ccsdg = Maxstyle().cuda()
    x = torch.randn([8,3,512,512]).cuda()
    x,sfs = ccsdg.encoder(x)
    out = ccsdg.img_decoder(x,sfs)
    print(out.size())


