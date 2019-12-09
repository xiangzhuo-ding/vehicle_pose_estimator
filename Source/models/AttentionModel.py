import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class EAUNet(nn.Module):
    '''Mixture of previous classes'''
    def __init__(self, n_classes, args):
        super(EAUNet, self).__init__()
        self.args = args
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        
        self.conv0 = double_conv(3, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 256)
        self.conv3 = double_conv(256, 512)
        self.conv4 = double_conv(512, 1024)

        self.mp = nn.MaxPool2d(2)
        
        self.Att1 = Attention_block(F_g=1280, F_l = 1024, F_int=512)
        self.Att2 = Attention_block(F_g=512, F_l = 512, F_int=256)

        self.up1 = up(1280 + 1024, 512)
        self.up2 = up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)


        
    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.mp(self.conv0(x))
        x2 = self.mp(self.conv1(x1))
        x3 = self.conv3(self.mp(self.conv2(x2)))
        x4 = self.conv4(self.mp(x3))

        IMG_WIDTH = 1024
        IMG_HEIGHT = IMG_WIDTH // 16 * 5
        x_center = x[:, :, :, IMG_WIDTH // 8: -IMG_WIDTH // 8]
        feats = self.base_model.extract_features(x_center)   
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8])
        if self.args.cuda:
            bg = bg.cuda()
        feats = torch.cat([bg, feats, bg], 3)
        
        # Add positional info
        p1d = (1, 1)
        feats = F.pad(feats, p1d, "constant", 0)
        u1 = self.up1.up(feats)
        x4 = self.Att1(g=u1, x=x4)
        u1 = self.up1(u1, x4)

        # Adding another attention
        u2 = self.up2.up(u1)
        x3 = self.Att2(g=u2, x=x3)
        u2 = self.up2(u2, x3)


        
        output = self.outc(u2)
        return output



class AttentionUnet(nn.Module):
    '''Mixture of previous classes'''
    def __init__(self, n_classes, args):
        super(AttentionUnet, self).__init__()
        self.args = args
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        
        self.conv0 = double_conv(3, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)

        self.mp = nn.MaxPool2d(2)
        
        self.Att1 = Attention_block(F_g=1280, F_l = 1024, F_int=512)
        self.Att2 = Attention_block(F_g=512, F_l = 512, F_int=256)

        self.up1 = up(1280 + 1024, 512)
        self.up2 = up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)


        
    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.mp(self.conv0(x))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        IMG_WIDTH = 1024
        IMG_HEIGHT = IMG_WIDTH // 16 * 5
        x_center = x[:, :, :, IMG_WIDTH // 8: -IMG_WIDTH // 8]
        feats = self.base_model.extract_features(x_center)   
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8])
        if self.args.cuda:
            bg = bg.cuda()
        feats = torch.cat([bg, feats, bg], 3)
        
        # Add positional info
        p1d = (1, 1)
        feats = F.pad(feats, p1d, "constant", 0)
        u1 = self.up1.up(feats)
        x4 = self.Att1(g=u1, x=x4)
        u1 = self.up1(u1, x4)

        # Adding another attention
        u2 = self.up2.up(u1)
        x3 = self.Att2(g=u2, x=x3)
        u2 = self.up2(u2, x3)
        output = self.outc(u2)
        return output


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi