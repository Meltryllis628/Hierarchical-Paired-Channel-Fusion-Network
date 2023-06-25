import torch.nn as nn
import torch


class RSA(nn.Module):
    def __init__(self, channel):
        super(RSA, self).__init__()
        self.channel = channel
        self.conv = nn.Conv2d(2, 1, (3, 3), padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_t0, f_t1, f_tield):
        f = f_t0 * f_t1
        max_f, _ = torch.max(f, dim=1, keepdim=True)
        mean_f = torch.mean(f, dim=1, keepdim=True)
        f = torch.cat((max_f, mean_f), dim=1)
        f = self.conv(f)
        m = self.sigmoid(f)
        m_r = m - 1
        f_hat = m_r * f_tield
        return f_hat
