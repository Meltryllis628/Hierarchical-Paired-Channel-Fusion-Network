import torch.nn as nn
import torch


class PCF(nn.Module):
    def __init__(self, channel, three_head=True):
        super(PCF, self).__init__()
        self.channel = channel
        self.three_head = three_head
        self.g_conv1 = nn.Conv2d(
            self.channel * 2,
            self.channel,
            (3, 3),
            dilation=1,
            groups=self.channel,
            padding=1,
        )
        self.g_conv2 = nn.Conv2d(
            self.channel * 2,
            self.channel,
            (3, 3),
            dilation=2,
            groups=self.channel,
            padding=2,
        )
        self.g_conv3 = nn.Conv2d(
            self.channel * 2,
            self.channel,
            (3, 3),
            dilation=3,
            groups=self.channel,
            padding=3,
        )
        self.g_conv4 = nn.Conv2d(
            self.channel * 2,
            self.channel,
            (3, 3),
            dilation=4,
            groups=self.channel,
            padding=4,
        )
        self.conv = nn.Conv2d(4 * self.channel, 2 * self.channel, (1, 1))

    def forward(self, f_t0, f_t1, f=0):
        x = torch.cat((f_t0, f_t1), dim=1)
        x_channels = torch.chunk(x, chunks=2*self.channel, dim=1)
        
        f_s = torch.cat(
            [
                x_channels[(i % 2) * self.channel + i // 2]
                for i in range(2*self.channel)
            ],
            dim=1,
        )
        
        f_s_1 = self.g_conv1(f_s)
        f_s_2 = self.g_conv2(f_s)
        f_s_3 = self.g_conv3(f_s)
        f_s_4 = self.g_conv4(f_s)
        f_s_ = torch.cat((f_s_1, f_s_2, f_s_3, f_s_4), dim=1)
        f_t = self.conv(f_s_)
        if self.three_head:
            f_tield = torch.cat((f_t, f), dim=1)
        else:
            f_tield = f_t
        return f_tield
