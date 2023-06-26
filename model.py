import torch
import torch.nn as nn
from modules.Conv import ConvBlock
from modules.MPFL import MPFL
from modules.PCF import PCF
from modules.RSA import RSA

NORMAL = 64
SMALL = 32
TINY = 16


class HPCFNet(nn.Module):
    def __init__(self, channels=SMALL) -> None:
        super(HPCFNet, self).__init__()
        self.conv_1_2_a = ConvBlock(3, channels, stride=1)
        self.conv_1_2_b = ConvBlock(3, channels, stride=1)
        self.conv_2_2_a = ConvBlock(channels, channels * 2)
        self.conv_2_2_b = ConvBlock(channels, channels * 2)
        self.conv_3_3_a = ConvBlock(channels * 2, channels * 4)
        self.conv_3_3_b = ConvBlock(channels * 2, channels * 4)
        self.conv_4_3_a = ConvBlock(channels * 4, channels * 8)
        self.conv_4_3_b = ConvBlock(channels * 4, channels * 8)
        self.conv_5_3_a = ConvBlock(channels * 8, channels * 8)
        self.conv_5_3_b = ConvBlock(channels * 8, channels * 8)
        self.pcf_1 = PCF(channels * 8, False)
        self.rsa_1 = RSA(channels * 8)
        self.conv_1 = ConvBlock(channels * 16, channels * 8, stride=1)
        self.mpfl_1 = MPFL(channels * 8)
        self.deconv_1 = nn.ConvTranspose2d(
            channels * 16, channels * 8, kernel_size=2, stride=2
        )
        self.pcf_2 = PCF(channels * 8, True)
        self.rsa_2 = RSA(channels * 8)
        self.conv_2 = ConvBlock(channels * 24, channels * 8, stride=1)
        self.mpfl_2 = MPFL(channels * 8)
        self.deconv_2 = nn.ConvTranspose2d(
            channels * 16, channels * 4, kernel_size=2, stride=2
        )
        self.pcf_3 = PCF(channels * 4, True)
        self.rsa_3 = RSA(channels * 4)
        self.conv_3 = ConvBlock(channels * 12, channels * 4, stride=1)
        self.mpfl_3 = MPFL(channels * 4)
        self.deconv_3 = nn.ConvTranspose2d(
            channels * 8, channels * 2, kernel_size=2, stride=2
        )
        self.pcf_4 = PCF(channels * 2, True)
        self.rsa_4 = RSA(channels * 2)
        self.conv_4 = ConvBlock(channels * 6, channels * 2, stride=1)
        self.mpfl_4 = MPFL(channels * 2)
        self.deconv_4 = nn.ConvTranspose2d(
            channels * 4, channels, kernel_size=2, stride=2
        )
        self.pcf_5 = PCF(channels, True)
        self.conv_5 = ConvBlock(channels * 3, channels, stride=1)
        self.conv_6 = ConvBlock(channels, 2, stride=1, kernel_size=1, padding=0)

    def forward(self, x):
        t0, t1 = torch.chunk(x, chunks=2, dim=2)
        t0_1 = self.conv_1_2_a(t0)
        t1_1 = self.conv_1_2_b(t1)
        t0_2 = self.conv_2_2_a(t0_1)
        t1_2 = self.conv_2_2_b(t1_1)
        t0_3 = self.conv_3_3_a(t0_2)
        t1_3 = self.conv_3_3_b(t1_2)
        t0_4 = self.conv_4_3_a(t0_3)
        t1_4 = self.conv_4_3_b(t1_3)
        t0_5 = self.conv_5_3_a(t0_4)
        t1_5 = self.conv_5_3_b(t1_4)
        f = self.pcf_1(t0_5, t1_5)
        f = self.rsa_1(t0_5, t1_5, f)
        f = self.conv_1(f)
        f = self.mpfl_1(f)
        f = self.deconv_1(f)
        f = self.pcf_2(t0_4, t1_4, f)
        f = self.rsa_2(t0_4, t0_4, f)
        f = self.conv_2(f)
        f = self.mpfl_2(f)
        f = self.deconv_2(f)
        f = self.pcf_3(t0_3, t1_3, f)
        f = self.rsa_3(t0_3, t1_3, f)
        f = self.conv_3(f)
        f = self.mpfl_3(f)
        f = self.deconv_3(f)
        f = self.pcf_4(t0_2, t1_2, f)
        f = self.rsa_4(t0_2, t1_2, f)
        f = self.conv_4(f)
        f = self.mpfl_4(f)
        f = self.deconv_4(f)
        f = self.pcf_5(t0_1, t1_1, f)
        f = self.conv_5(f)
        f = self.conv_6(f)
        return f
