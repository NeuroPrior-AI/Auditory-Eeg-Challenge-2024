import torch
import torch.nn as nn

class EEGEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super(EEGEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        return x

# just put shape
eeg_encoder = EEGEncoder(in_channels=channels, out_channels=64, kernel_size=3, pool_size=2)

eeg_data = torch.randn((batch_size, channels, segment_length))

output = eeg_encoder(eeg_data)
print("Encoder Output Shape:", output.shape)
