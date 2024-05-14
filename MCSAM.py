import torch
import torch.nn as nn


class MCSAM(torch.nn.Module):
    def __init__(self, inChannels, last_channel):  # inChannels表输入通道数
        super(MCSAM, self).__init__()
        # 第一层池化
        self.branch1 = nn.AvgPool3d(kernel_size=1)

        self.branch2 = nn.MaxPool3d(kernel_size=1)

        self.branch3_1 = nn.Conv3d(in_channels=inChannels, out_channels=last_channel, kernel_size=3, padding=1)
        self.branch3_2 = nn.Conv3d(last_channel, last_channel, kernel_size=3, padding=1)

        self.branch4_1 = nn.Conv3d(inChannels, last_channel, kernel_size=1)
        self.branch4_2 = nn.Conv3d(last_channel, last_channel, kernel_size=1)
        self.a = nn.Conv3d(inChannels*4, last_channel, kernel_size=3, padding=1)
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3_1 = self.branch3_1(x)
        branch3 = self.branch3_2(branch3_1)
        branch4_1 = self.branch4_1(x)
        branch4 = self.branch4_2(branch4_1)
        output = [branch2, branch3, branch4, branch1]
        output = torch.cat(output, dim=1)
        return self.a(output)