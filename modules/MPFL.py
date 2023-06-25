import torch
import torch.nn as nn


class MPFL(nn.Module):
    def __init__(self, channel):
        super(MPFL, self).__init__()
        self.conv_a_1 = nn.Conv2d(channel, channel // 2, (1, 1))
        self.conv_a_2 = nn.Conv2d(channel, channel // 2, (1, 1))
        self.conv_a_3 = nn.Conv2d(channel, channel // 2, (1, 1))
        self.conv_a_4 = nn.Conv2d(channel, channel // 2, (1, 1))
        self.conv_b_1 = nn.Conv2d(channel, channel // 2, (3, 1), padding=(1, 0))
        self.conv_b_2 = nn.Conv2d(channel, channel // 2, (3, 1), padding=(1, 0))
        self.conv_c_1 = nn.Conv2d(channel, channel // 2, (1, 3), padding=(0, 1))
        self.conv_c_2 = nn.Conv2d(channel, channel // 2, (1, 3), padding=(0, 1))
        self.conv_d = nn.Conv2d(channel, channel // 2, (3, 3), padding=(1, 1))

    def forward(self, x):
        x_1_ = torch.chunk(x, chunks=2, dim=2)
        x_1 = list(torch.chunk(x_1_[0], chunks=2, dim=3))
        x_1.extend(torch.chunk(x_1_[1], chunks=2, dim=3))
        x_2 = list(torch.chunk(x, chunks=2, dim=2))
        x_3 = list(torch.chunk(x, chunks=2, dim=3))
        x_1[0] = self.conv_a_1(x_1[0])
        x_1[1] = self.conv_a_2(x_1[1])
        x_1[2] = self.conv_a_3(x_1[2])
        x_1[3] = self.conv_a_4(x_1[3])
        x_2[0] = self.conv_b_1(x_2[0])
        x_2[1] = self.conv_b_2(x_2[1])
        x_3[0] = self.conv_c_1(x_3[0])
        x_3[1] = self.conv_c_2(x_3[1])
        x_4 = self.conv_d(x)
        x_1_close_2 = torch.cat((x_1[2], x_1[3]), dim=3)
        x_1_close_1 = torch.cat((x_1[0], x_1[1]), dim=3)
        x_1_close = torch.cat((x_1_close_1, x_1_close_2), dim=2)
        x_2_close = torch.cat((x_2[0], x_2[1]), dim=2)
        x_3_close = torch.cat((x_3[0], x_3[1]), dim=3)
        out = torch.cat((x_1_close, x_2_close, x_3_close, x_4), dim=1)
        return out
