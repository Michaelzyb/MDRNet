import torch
import torch.nn as nn
import torch.nn.functional as F
class conv_stem(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(conv_stem, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(),
            #nn.Dropout(0.3),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.double_conv(x)

class down_conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(down_conv, self).__init__()
        self.down = nn.MaxPool2d(2, 2)
        self.conv = conv_stem(in_channel, out_channel)
    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_channel, output_channel, bilinear = False):
        super(up_conv, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Conv2d(in_channel, in_channel//2, 1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channel, in_channel//2, 2, 2)
        self.conv = conv_stem(in_channel, output_channel)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        batch_size, channels, _, _ = x.size()

        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        max_pool = F.adaptive_max_pool2d(x, 1).view(batch_size, channels)

        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))

        gate = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1)

        return x * gate.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super(SpatialAttention, self).__init__()

        # 卷积层，用于计算空间注意力
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, channels, height, width]

        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, height, width]
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [batch_size, 1, height, width]

        x_cat = torch.cat((avg_pool, max_pool), dim=1)  # [batch_size, 2, height, width]

        attention_map = self.conv(x_cat)  # [batch_size, 1, height, width]

        attention_map = self.sigmoid(attention_map)  # [batch_size, 1, height, width]

        return x * attention_map  # [batch_size, channels, height, width]

class attention(nn.Module):
    def __init__(self, in_channel, reduction):
        super(attention, self).__init__()
        self.ca = ChannelAttention(in_channel, reduction)
        self.sa = SpatialAttention(3)

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out
class abunet(nn.Module):
    def __init__(self, in_channel, num_class):
        super(abunet, self).__init__()
        channels = [64, 128, 256, 512, 1024]
        self.conv1 = conv_stem(in_channel, channels[0])
        self.d1 = down_conv(channels[0], channels[1])
        self.d2 = down_conv(channels[1], channels[2])
        self.d3 = down_conv(channels[2], channels[3])
        self.d4 = down_conv(channels[3], channels[4])
        self.u1 = up_conv(channels[4], channels[3])
        self.u2 = up_conv(channels[3], channels[2])
        self.u3 = up_conv(channels[2], channels[1])
        self.u4 = up_conv(channels[1], channels[0])
        self.conv2 = nn.Conv2d(channels[0], num_class, 3, 1, 1)
        self.attention1 = attention(64, 16)
        self.attention2 = attention(128, 16)
        self.attention3 = attention(256, 16)
        self.attention4 = attention(512, 16)
    def forward(self, x):
        x1 = self.conv1(x)
        x1_1 = self.attention1(x1)
        x2 = self.d1(x1_1)
        x2_2 = self.attention2(x2)
        x3 = self.d2(x2_2)
        x3_3 = self.attention3(x3)
        x4 = self.d3(x3_3)
        x4_4 = self.attention4(x4)
        x5 = self.d4(x4_4)
        x6 = self.u1(x5, x4)
        x7 = self.u2(x6, x3)
        x8 = self.u3(x7, x2)
        x9 = self.u4(x8, x1)
        x10 = self.conv2(x9)
        return x10
