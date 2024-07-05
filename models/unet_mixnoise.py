from torch import nn
import torch
import torch.nn.functional as F
# from models.unet import UnetBlock, SaveFeatures
# from models.resnet import resnet34_mix, resnet18, resnet50, resnet101, resnet152
from models.unet import UnetBlock, SaveFeatures
from models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
from models.mix import *
from models.noise_encoder import NoiseEncoder

class UNet_Mix(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2,ini=3):
        super().__init__()
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

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        self.generate_img = nn.ConvTranspose2d(256, 3, 2, stride=2)

    def encoder(self,x):
        x, sfs = self.res(x)

        return x,sfs

    def forward(self, x):
        x, sfs = self.res(x)

        channel_prompt_onehot = torch.softmax(self.channel_prompt/0.1, dim=0)
        f_content = sfs[0] * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        f_style = sfs[0] * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)
        # x = self.first_layer(x)
        # channel_prompt_onehot = torch.softmax(self.channel_prompt / tau, dim=0)
        # f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)

        x = F.relu(x)
        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        output = self.up5(x)
        # output = F.softmax(output)
        #
        # threshold = 0.5
        # binary_segmentations = []
        # for channel in range(self.num_classes):
        #     binary_segmentation = (output[:, channel, :, :] > threshold).float()
        #     binary_segmentations.append(binary_segmentation)

        return output

    def close(self):
        for sf in self.sfs: sf.remove()

class generater(nn.Module):
    def __init__(self, num_classes=2, mixstyle_layers=['layer2'],random_type='MixStyle',ini=3,out=3,batch=4):
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
        self.generate_img = nn.ConvTranspose2d(256, out, 2, stride=2)

        if mixstyle_layers:
            if random_type == 'MixNoise':
                self.random = MixNoise(batch_size=batch,p=0.5)
            elif random_type == 'TriD':
                self.random = TriD(p=0.5)
            elif random_type == 'MixStyle':
                self.random = MixStyle(p=0.5, mix='random')
            elif random_type == 'EFDMixStyle':
                self.random = EFDMix(p=0.5, mix='random')
            else:
                raise ValueError('The random method type is wrong!')
            print('Insert Random Style after the following layers: {}'.format(mixstyle_layers))

    def forward(self, x,sfs, data):
        noise = torch.randn_like(data)
        [gamma_noise1,gamma_noise2,gamma_noise3,gamma_noise4,gamma_noise5,
         beta_noise1, beta_noise2, beta_noise3, beta_noise4, beta_noise5] = self.noiseEncoder(noise)

        # sfs3,sfs2,sfs1,sfs0 = sfs[3].features,sfs[2].features,sfs[1].features,sfs[0].features

        x = F.relu(x)
        if 'layer0' in self.mixstyle_layers:
            #sfs[3] = self.random(sfs[3],gamma_noise4,beta_noise4)
            sfs[3] = self.random(sfs[3])
        x = self.up1(x, sfs[3])

        if 'layer1' in self.mixstyle_layers:
            #sfs[2] = self.random(sfs[2],gamma_noise3,beta_noise3)
            sfs[2] = self.random(sfs[2])
        x = self.up2(x, sfs[2])

        if 'layer2' in self.mixstyle_layers:
            #sfs[1] = self.random(sfs[1],gamma_noise2,beta_noise2)
            sfs[1] = self.random(sfs[1])
        x = self.up3(x, sfs[1])

        if 'layer3' in self.mixstyle_layers:
            #sfs[0] = self.random(sfs[0], gamma_noise1,beta_noise1)
            sfs[0] = self.random(sfs[0])
        x = self.up4(x, sfs[0])

        output = self.generate_img(x)
        return output

if __name__ == '__main__':
    ccsdg = UNet_Mix().cuda()
    x = torch.randn([8,3,512,512]).cuda()
    out = ccsdg.generate(x)
    print('out ',out.size())
