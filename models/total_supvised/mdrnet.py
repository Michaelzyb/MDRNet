import time
import torch
from torch import nn
import torch.nn.functional as F
class conv_stem(nn.Module):#1.Convolutional Stem Module
    def __init__(self, in_channels, out_channels):
        super(conv_stem, self).__init__()
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        ))
        self.layer = nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class MultiScaleChannelAttention_1(nn.Module):#Multiscale attention of the first two hybrid transformer layers
    def __init__(self, in_channels):
        super(MultiScaleChannelAttention_1, self).__init__()

        # Convolutional layers at three different scales
        self.conv3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=4, padding=3, dilation=2)
        self.conv11 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=6, padding=5, dilation=2)

        # Fully Connected Layer of Channel Attention Mechanisms
        self.fc1 = nn.Linear(in_channels, in_channels // 4)
        self.fc2 = nn.Linear(in_channels // 4, in_channels)

    def forward(self, x):
        # Processing inputs using different scales of convolution
        x3 = self.conv3(x)  # C/2
        x7 = self.conv7(x)  # C/4
        x11 = self.conv11(x)  # C/4

        x_concat = torch.cat((x3, x7, x11), dim=1)

        b, c, h, w = x_concat.size()
        avg_pool = F.adaptive_avg_pool2d(x_concat, 1).view(b, c)

        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))

        out = torch.sigmoid(avg_out).view(b, c, 1, 1)

        x_out = x_concat * out.expand_as(x_concat)

        return x_out

class MultiScaleChannelAttention_2(nn.Module):  # Multiscale attention of the third hybrid transformer layer
    def __init__(self, in_channels):
        super(MultiScaleChannelAttention_2, self).__init__()

        self.conv3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=4, padding=3, dilation=2)

        self.fc1 = nn.Linear(in_channels, in_channels // 4)
        self.fc2 = nn.Linear(in_channels // 4, in_channels)

    def forward(self, x):
        x3 = self.conv3(x)  # C/2
        x7 = self.conv7(x)  # C/2

        x_concat = torch.cat((x3, x7), dim=1)

        b, c, h, w = x_concat.size()
        avg_pool = F.adaptive_avg_pool2d(x_concat, 1).view(b, c)

        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))

        out = torch.sigmoid(avg_out).view(b, c, 1, 1)

        x_out = x_concat * out.expand_as(x_concat)

        return x_out

class Img2Seq(nn.Module):
    def __init__(self):
        super(Img2Seq, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x = x.permute(0, 2, 1)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        N, seq_length, embed_dim = x.shape

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, seq_length, head_dim)
        keys = keys.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, seq_length, head_dim)
        queries = queries.view(N, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, seq_length, head_dim)

        energy = torch.einsum("nhqd,nhkd->nhqk", queries, keys)  # (N, num_heads, seq_length, seq_length)
        attention = F.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nhld->nhqd", attention, values).transpose(1, 2).contiguous()  # (N, seq_length, num_heads, head_dim)
        out = out.view(N, seq_length, embed_dim)  # (N, seq_length, embed_dim)

        return out

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim)

    def forward(self, x):
        # LayerNorm first
        x_norm = self.norm1(x)
        attention = self.attention(x_norm)
        x = attention + x
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x)
        return ffn_out + x_norm2

class Seq2Img(nn.Module):
    def __init__(self, height, width):
        super(Seq2Img, self).__init__()
        self.height = height
        self.width = width

    def forward(self, x):
        B, HW, C = x.shape
        x = x.permute(0, 2, 1)
        x = x.view(B, C, self.height, self.width)
        return x

class MSCA_Transformer_Module(nn.Module):#2.MSCA Transformer Module
    def __init__(self, in_channels, height, width, num_heads, Transformer_Block_num):#in_channels(embed_dim)
        super(MSCA_Transformer_Module, self).__init__()
        self.MultiScaleChannelAttention = MultiScaleChannelAttention_1(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.Img2seq = Img2Seq()
        self.TransformerBlock = TransformerBlock(in_channels, num_heads)
        self.Seq2Img = Seq2Img(height, width)
        self.Block_num = Transformer_Block_num

    def forward(self, x):
        x_1 = self.conv(x)
        x_2 = self.MultiScaleChannelAttention(x)
        x = x_1+x_2
        x = self.Img2seq(x)
        z = self.Block_num
        for i in range(z):
            x = self.TransformerBlock(x)
        out = self.Seq2Img(x)
        return out

class MSCA_Transformer_Module_1(nn.Module):#3.The last two MSCA Transformer modules
    def __init__(self, in_channels, height, width, num_heads, Transformer_Block_num):#in_channels(embed_dim)
        super(MSCA_Transformer_Module_1, self).__init__()
        self.MultiScaleChannelAttention = MultiScaleChannelAttention_2(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.Img2seq = Img2Seq()
        self.TransformerBlock = TransformerBlock(in_channels, num_heads)
        self.Seq2Img = Seq2Img(height, width)
        self.Block_num = Transformer_Block_num

    def forward(self, x):
        x_1 = self.conv(x)
        x_2 = self.MultiScaleChannelAttention(x)
        x = x_1+x_2
        x = self.Img2seq(x)
        z = self.Block_num
        for i in range(z):
            x = self.TransformerBlock(x)
        out = self.Seq2Img(x)
        return out

class PatchMerging(nn.Module):#4.Patch Merging Module
    def __init__(self, embed_dim, out_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduction = nn.Linear(4 * embed_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(4 * embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"Height ({H}) and width ({W}) must be even."
        # Reshape and select patches
        x0 = x[:, :, 0::2, 0::2]  # B, C, H/2, W/2
        x1 = x[:, :, 1::2, 0::2]  # B, C, H/2, W/2
        x2 = x[:, :, 0::2, 1::2]  # B, C, H/2, W/2
        x3 = x[:, :, 1::2, 1::2]  # B, C, H/2, W/2

        # Concatenate along the channel dimension
        x = torch.cat([x0, x1, x2, x3], 1)  # B, 4*C, H/2, W/2

        # Reshape to merge the channels
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, 4 * C)  # B, H/2*W/2, 4*C
        x = self.norm(x)  # Normalization
        x = self.reduction(x)  # Channel reduction

        # Reshape back to (B, Dim, H/2, W/2)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, -1, H // 2, W // 2)
        return x

def shuffle_chnls(x, groups=8):
    """Channel Shuffle"""

    bs, chnls, h, w = x.data.size()

    # If the number of channels is not a grouped integer being, the Channel Shuffle operation cannot be performed and returns x directly
    if chnls % groups:
        raise AttributeError('Please confirm channels can be exact division!')

    # Calculate the number of channels in a group for Channel Shuffle.
    chnls_per_group = chnls // groups

    # Perform a channel shuffle operation, don't use the view directly into 5 dimensions, the exported onnx will report an error
    x = x.unsqueeze(1)
    x = x.view(bs, groups, chnls_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bs, -1, h, w)

    return x

class DenselyShuffleMerge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = shuffle_chnls(x, 8)
        x = self.conv(x)
        return x
class DenseShuffleConv_1(nn.Module):#5.Dense shuffle group convolution
    def __init__(self):
        super(DenseShuffleConv_1, self).__init__()

        # Upper sampling layer 28x28
        self.up_conv_layer1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Upper sampling layer 14x14
        self.up_conv_layer2_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_conv_layer2_2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Convolutional layers for channel reorganization
        self.group_convs = DenselyShuffleMerge(128, 64)

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(64, 32, kernel_size=1)

    def forward(self, x1, x2, x3):
        x2_up = self.up_conv_layer1(x2)

        x3_up = self.up_conv_layer2_1(x3)
        x3_up = self.up_conv_layer2_2(x3_up)

        x_concat = torch.cat((x1, x2_up, x3_up), dim=1)
        group_outputs = self.group_convs(x_concat)

        output = self.final_conv(group_outputs)

        return output
class DenseShuffleConv_2(nn.Module):
    def __init__(self):
        super(DenseShuffleConv_2, self).__init__()

        self.down_conv_layer = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.up_conv_layer = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.group_convs = DenselyShuffleMerge(128, 64)

        self.final_conv = nn.Conv2d(64, 32, kernel_size=1)

    def forward(self, x1, x2, x3):

        x1_down = self.down_conv_layer(x1)

        x3_up = self.up_conv_layer(x3)

        x_concat = torch.cat((x1_down, x2, x3_up), dim=1)
        group_outputs = self.group_convs(x_concat)

        output = self.final_conv(group_outputs)

        return output

class DenseShuffleConv_3(nn.Module):
    def __init__(self):
        super(DenseShuffleConv_3, self).__init__()

        self.down_conv_layer1_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.down_conv_layer1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.down_conv_layer2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.group_convs = DenselyShuffleMerge(128, 64)

        self.final_conv = nn.Conv2d(64, 32, kernel_size=1)

    def forward(self, x1, x2, x3):

        x1_down = self.down_conv_layer1_1(x1)
        x1_down = self.down_conv_layer1_2(x1_down)

        x2_down = self.down_conv_layer2(x2)

        x_concat = torch.cat((x1_down, x2_down, x3), dim=1)
        group_outputs = self.group_convs(x_concat)

        output = self.final_conv(group_outputs)

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=(2, 3))  # (batch_size, channels)
        avg_out = self.fc1(avg_out)
        avg_out = F.relu(avg_out)
        avg_out = self.fc2(avg_out)
        avg_out = torch.sigmoid(avg_out).view(x.size(0), x.size(1), 1, 1)

        return x * avg_out.expand_as(x)

class RegionAttentionModule(nn.Module): #6.RegionAttentionModule
    def __init__(self, in_channels, grid_size):
        super(RegionAttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.grid_size = grid_size

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        grid_size = self.grid_size

        # Make sure the height and width are divisible by grid_size.
        assert height % grid_size == 0 and width % grid_size == 0, "Height and width must be divisible by grid_size."

        # Number of counting areas
        region_count_height = height // grid_size
        region_count_width = width // grid_size

        # Reshape x to (batch_size, channels, region_count_height, grid_size, region_count_width, grid_size)
        # Then transpose it to (batch_size, region_count_height, region_count_width, channels, grid_size, grid_size)
        x = x.view(batch_size, channels, region_count_height, grid_size, region_count_width, grid_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (batch_size, region_count_height, region_count_width, channels, grid_size, grid_size)

        # Merging the grid_size dimension to apply channel attention
        x = x.view(-1, channels, grid_size, grid_size)  # (batch_size * region_count_height * region_count_width, channels, grid_size, grid_size)

        # Applied Channel Attention
        region_attention = self.channel_attention(x)

        # Reshape back to its original shape
        out = region_attention.view(batch_size, region_count_height, region_count_width, channels, grid_size, grid_size)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()  # (batch_size, channels, region_count_height, grid_size, region_count_width, grid_size)

        # Merge back to full size
        out = out.view(batch_size, channels, height, width)  # (batch_size, channels, height, width)

        return out


class RegionAttention(nn.Module):
    def __init__(self, in_channels):
        super(RegionAttention, self).__init__()
        self.region_attention_4x4 = RegionAttentionModule(in_channels, 4)
        self.region_attention_8x8 = RegionAttentionModule(in_channels, 8)

    def forward(self, x):
        out_4x4 = self.region_attention_4x4(x)
        out_8x8 = self.region_attention_8x8(x)

        out = out_4x4 + out_8x8
        return out

class MDR_Net(nn.Module):
    def __init__(self, in_c, num_classes):
        super(HDR_Net, self).__init__()
        self.conv_stem = conv_stem(in_c, 32)
        self.MSCA_Transformer_Module_1 = MSCA_Transformer_Module(32, 56, 56, 4, 1)#Adjust height and width according to input image size
        self.MSCA_Transformer_Module_2 = MSCA_Transformer_Module_1(32, 28, 28, 4, 2)#Adjust height and width according to input image size
        self.MSCA_Transformer_Module_3 = MSCA_Transformer_Module_1(64, 14, 14, 4, 3)#Adjust height and width according to input image size
        self.Patch_Merging_1 = PatchMerging(32, 32)
        self.Patch_Merging_2 = PatchMerging(32, 64)
        self.DSGC_1 = DenseShuffleConv_1()
        self.DSGC_2 = DenseShuffleConv_2()
        self.DSGC_3 = DenseShuffleConv_3()
        self.conv1 = nn.Conv2d(32, 12, 1)
        self.conv2 = nn.Conv2d(32, 12, 1)
        self.conv3 = nn.Conv2d(32, 12, 1)
        self.up_conv_layer_x2 = nn.ConvTranspose2d(12, 12, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_conv_layer_x4_1 = nn.ConvTranspose2d(12, 12, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_conv_layer_x4_2 = nn.ConvTranspose2d(12, 12, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.RegionAttentionModule = RegionAttention(36)
        self.conv1x1 = nn.Conv2d(36, num_classes, 1)
        self.upx4_1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upx4_2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, input_tensor):
        output_tensor = self.conv_stem(input_tensor)
        output_tensor_1 = self.MSCA_Transformer_Module_1(output_tensor)
        output_tensor_2 = self.Patch_Merging_1(output_tensor_1)
        output_tensor_2 = self.MSCA_Transformer_Module_2(output_tensor_2)
        output_tensor_3 = self.Patch_Merging_2(output_tensor_2)
        output_tensor_3 = self.MSCA_Transformer_Module_3(output_tensor_3)
        y_1 = self.conv1(self.DSGC_1(output_tensor_1, output_tensor_2, output_tensor_3))
        y_2 = self.conv2(self.DSGC_2(output_tensor_1, output_tensor_2, output_tensor_3))
        y_3 = self.conv3(self.DSGC_3(output_tensor_1, output_tensor_2, output_tensor_3))
        y_2 = self.up_conv_layer_x2(y_2)
        y_3 = self.up_conv_layer_x4_2(self.up_conv_layer_x4_1(y_3))
        out = torch.cat((y_1, y_2, y_3), dim=1)
        out = self.RegionAttentionModule(out)
        out = self.conv1x1(out)
        out = self.upx4_2(self.upx4_1(out))
        return out
