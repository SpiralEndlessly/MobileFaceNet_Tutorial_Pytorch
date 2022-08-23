#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 11:37:25 2022
Split retrained model into front and back parts

@author: SpiralEndlessly
"""

from face_model import Conv_block, Depth_Wise, Residual
import torch
import torchvision.transforms as transforms
from torch import nn
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FrontEndNet(nn.Module):
    def __init__(self):
        super(FrontEndNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        
        return out

class BackEndNet(nn.Module):
    def __init__(self, latent_size):
        super(BackEndNet, self).__init__()
        self.conv_sep = nn.Conv2d(in_channels=128, out_channels=latent_size, kernel_size=(1,1), stride=(1,1), padding=(0,0), groups=1, bias=False)
        self.bn_sep = nn.BatchNorm2d(latent_size)
        self.relu = nn.ReLU()
        self.conv_dw = nn.Conv2d(latent_size, out_channels=latent_size, kernel_size=(7,7), stride=(1,1), padding=(0,0), groups=latent_size, bias=False)
        self.bn_dw = nn.BatchNorm2d(latent_size)
        self.conv_fin = nn.Conv2d(latent_size, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0), groups=1, bias=False)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        out = self.conv_sep(x)
        out = self.bn_sep(out)
        out = self.relu(out)
        out = self.conv_dw(out)
        out = self.bn_dw(out)
        out = self.conv_fin(out)
        
        return self.flatten(out)

def LoadPartialStateDict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
                continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

if __name__ == '__main__':
    state_dict = torch.load('Weights/Retrained', map_location=lambda storage, loc: storage)["net_state_dict"]
    front = FrontEndNet()
    back = BackEndNet(latent_size=320)

    LoadPartialStateDict(front, state_dict)
    LoadPartialStateDict(back, state_dict)

    dummy_input = torch.randn(1, 3, 112, 112, device=device)
    torch.onnx.export(front, dummy_input, "MobileFaceNet_Front.onnx", input_names=["input"], output_names=["output"], do_constant_folding=True, opset_version=14)

    dummy_input = torch.randn(1, 128, 7, 7, device=device)
    torch.onnx.export(back, dummy_input, "MobileFaceNet_ReducedBack.onnx", input_names=["input"], output_names=["output"], do_constant_folding=True, opset_version=14)