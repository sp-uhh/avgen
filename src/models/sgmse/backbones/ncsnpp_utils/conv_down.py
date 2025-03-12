import torch
import torch.nn as nn

class Conv_Down(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(Conv_Down, self).__init__()

        self.out_channels = out_channels
        channels = torch.linspace(in_channels, out_channels*4, 5, dtype=torch.int)

        self.conv1 = nn.Conv1d(channels[0], channels[1], kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv1d(channels[1], channels[2], kernel_size=kernel_size, stride=stride)
        self.conv3 = nn.Conv1d(channels[2], channels[3], kernel_size=kernel_size, stride=stride)
        self.conv4 = nn.Conv1d(channels[3], channels[4], kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        h = x.squeeze(2)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = h.unsqueeze(2)
        h = h.view(x.shape[0], self.out_channels, 4, 4)
        return h
