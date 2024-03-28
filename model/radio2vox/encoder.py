'''
Descripttion: 
version: 
Author: ZHIHA
Date: 2024-03-28 20:23:25
LastEditors: ZHIHA
LastEditTime: 2024-03-28 20:24:09
'''
import sys 
sys.path.append("/home/yzw/ai-wireless-sensing/src/models/radio2vox/")
import torch 
import torch.nn as nn
import torchvision.models


class Encoder(torch.nn.Module):
    def __init__(self,
        inp_channels=2,
        dim = 16) -> None:
        super(Encoder,self).__init__()
        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        self.pixel_att1 = PixelAwareAttention(dim * 3, dim * 3, 3, 3)
        self.pixel_att2 = PixelAwareAttention(dim * 3, dim * 3, 5, 5)
        self.pixel_att3 = PixelAwareAttention(dim * 3, dim * 3, 7, 7)
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(dim * 12, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        
    
    def forward(self, x):
        inp_patch1 = self.patch_embed(x)

        pixel_att1 = self.pixel_att1(inp_patch1)
        pixel_att2 = self.pixel_att2(inp_patch1)
        pixel_att3 = self.pixel_att3(inp_patch1)
        pixel_fusion = torch.cat([inp_patch1,pixel_att1,pixel_att2,pixel_att3],dim=1)

        features = self.layer1(pixel_fusion)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        
        return features
    
class PixelAwareAttention(nn.Module):
    def __init__(self, in_channels, out_channels, width_kernel_size, height_kernel_size):
        super(PixelAwareAttention, self).__init__()
        
        self.width_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, width_kernel_size), padding=(0, width_kernel_size // 2), bias=False)
        
        self.height_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(height_kernel_size, 1), padding=(height_kernel_size // 2, 0), bias=False)
        
        self.fuse_conv = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1, bias=False)
        
        self.attention = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        
        width_features = self.width_conv(x)
        height_features = self.height_conv(x)
    
        combined_features = torch.cat([width_features, height_features], dim=1)
        
        attention_weights = torch.sigmoid(self.attention(combined_features))
        
        fused_features = self.fuse_conv(combined_features)
        
        output = fused_features * attention_weights
        
        return output
    
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj1 = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.proj2 = nn.Conv2d(in_c, embed_dim, kernel_size=5, stride=1, padding=2, bias=bias)
        self.proj3 = nn.Conv2d(in_c, embed_dim, kernel_size=7, stride=1, padding=3, bias=bias)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((128, 128))
        
    def forward(self, x):
        x1 = self.proj1(x)
        x2 = self.proj2(x)
        x3 = self.proj3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.adaptive_pool(x)
        return x