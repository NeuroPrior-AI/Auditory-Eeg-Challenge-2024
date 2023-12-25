import torch
from transformers import Wav2Vec2Model
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from task2_regression.models.utils import init_weights, get_padding

LRELU_SLOPE = 0.1
NUM_MEL_BINS = 10  # Modify this based on your mel-spectrogram configuration

class ResBlock1(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], 
                   padding=get_padding(kernel_size, dilation[0])),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], 
                   padding=get_padding(kernel_size, dilation[1])),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], 
                   padding=get_padding(kernel_size, dilation[2]))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], 
                   padding=get_padding(kernel_size, dilation[0])),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], 
                   padding=get_padding(kernel_size, dilation[1]))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)
            

# class EEGtoMelAutoEncoder(nn.Module):
#     def __init__(self, h, latent_dim=64):
#         super(EEGtoMelAutoEncoder, self).__init__()
#         self.h = h
#         self.num_kernels = len(h.resblock_kernel_sizes)
#         self.num_upsamples = len(h.upsample_rates)

#         # Initial convolution layer adapted for EEG data
#         self.conv_pre = weight_norm(Conv2d(64, latent_dim, (4, 4), 1, padding=(2, 2)))

#         # ResBlock selection based on configuration
#         resblock = ResBlock1 if h.resblock == '1' else ResBlock2

#         # Upsampling layers initialization
#         # self.ups = nn.ModuleList()
#         # for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
#         #     # Ensure the dimensions are correct
#         #     self.ups.append(weight_norm(
#         #         ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
#         #                         k, u, padding=(k-u)//2)))
#         self.ups = nn.ModuleList()
#         for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
#             in_channels = h.upsample_initial_channel // (2 ** i)
#             out_channels = h.upsample_initial_channel // (2 ** (i + 1))
#             # Verify these dimensions are correct for your model
#             layer = ConvTranspose1d(in_channels, out_channels, k, u, padding=(k - u) // 2)
#             self.ups.append(weight_norm(layer))

#         # Residual blocks initialization
#         self.resblocks = nn.ModuleList()
#         for i in range(len(self.ups)):
#             ch = h.upsample_initial_channel//(2**(i+1))
#             for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
#                 self.resblocks.append(resblock(h, ch, k, d))

#         # Final convolutional layer
#         self.conv_post = weight_norm(Conv1d(ch, NUM_MEL_BINS, 7, 1, padding=3))

#         # Initialize weights
#         self.ups.apply(init_weights)
#         self.resblocks.apply(init_weights)
#         self.conv_post.apply(init_weights)

#     def forward(self, x):
#         # Reshape EEG data to fit Conv2d (batch_size, channels, height, width)
#         x = x.reshape(-1, 64, 30, 64)  # Assuming EEG data is reshaped appropriately

#         # Initial convolution
#         x = self.conv_pre(x)
#         x = x.reshape(x.size(0), -1, x.size(-1))  # Flatten for Conv1d

#         # Sequentially apply upsampling and residual blocks
#         for i in range(self.num_upsamples):
#             x = F.leaky_relu(x, LRELU_SLOPE)
#             x = self.ups[i](x)
#             xs = None
#             for j in range(self.num_kernels):
#                 if xs is None:
#                     xs = self.resblocks[i*self.num_kernels+j](x)
#                 else:
#                     xs += self.resblocks[i*self.num_kernels+j](x)
#             x = xs / self.num_kernels

#         # Apply final leaky ReLU and convolution
#         x = F.leaky_relu(x)
#         x = self.conv_post(x)

#         # Tanh activation to normalize output
#         x = torch.tanh(x)

#         return x

#     def remove_weight_norm(self):
#         # Function to remove weight normalization from all layers
#         print('Removing weight norm...')
#         for l in self.ups:
#             remove_weight_norm(l)
#         for l in self.resblocks:
#             l.remove_weight_norm()
#         remove_weight_norm(self.conv_pre)
#         remove_weight_norm(self.conv_post)

class EEGtoMelAutoEncoder(nn.Module):
    def __init__(self, h, latent_dim=64):
        super(EEGtoMelAutoEncoder, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = weight_norm(Conv1d(latent_dim, latent_dim, 5, 1, padding=2))
        # self.conv_pre = Conv2d(64, latent_dim, (4, 4), 1, padding=(2, 2))
        
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            in_channels = h.upsample_initial_channel // (2 ** i)
            out_channels = h.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(ConvTranspose1d(in_channels, out_channels, k, u, padding=(k - u) // 2))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = Conv1d(ch, NUM_MEL_BINS, 4, 4, padding=0)

        self.ups.apply(init_weights)
        self.resblocks.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        # print(f"[DEBUG] Input shape: {x.shape}")

        # Reshape EEG data to fit Conv2d (batch_size, channels, height, width)
        # x = x.reshape(-1, 64, 30, 64)
        # print(f"[DEBUG] After reshape for Conv2d: {x.shape}")

        x = self.conv_pre(x)
        # print(f"[DEBUG] After conv_pre: {x.shape}")

        # Adjust flattening logic for Conv1d layers
        # Ensure the number of channels is correct for the ConvTranspose1d layers
        # x = x.permute(0, 2, 3, 1)  # Change the order of dimensions
        # x = x.reshape(x.size(0), x.size(1), -1)  # Correctly flatten
        # print(f"[DEBUG] After flatten for Conv1d: {x.shape}")

        for i, upsample in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = upsample(x)
            # print(f"[DEBUG] After upsampling {i}, shape: {x.shape}")
            # Ensure the dimensions are compatible with ResBlock input
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels

        # Apply final leaky ReLU and convolution
        x = F.leaky_relu(x)
        # print(f"[DEBUG] After residual blocks, shape: {x.shape}")
        x = self.conv_post(x)
        # print(f"[DEBUG] After conv_post, shape: {x.shape}")

        # Tanh activation to normalize output
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        # Function to remove weight normalization from all layers
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)