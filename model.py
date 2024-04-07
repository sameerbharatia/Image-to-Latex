import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding2D, self).__init__()
        self.register_buffer('pe', self._get_pe(d_model, height, width))

    @staticmethod
    def _get_pe(d_model, height, width):
        pe = torch.zeros(d_model, height, width)
        position_h = torch.arange(height).unsqueeze(1)
        position_w = torch.arange(width).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[0:d_model:2, :, :] = torch.sin(position_h * div_term).transpose(0, 1).unsqueeze(2)
        pe[1:d_model:2, :, :] = torch.cos(position_w * div_term).transpose(0, 1).unsqueeze(1)

        return pe

    def forward(self, x):
        return x + self.pe[:, :x.size(2), :x.size(3)]

# class Bottleneck(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(Bottleneck, self).__init__()
#         mid_channels = out_channels // 4
#         # The bottleneck consists of three convolutions: 1x1, 3x3, and 1x1. This is a simplified version.
#         self.reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False) # Reduce dimension
#         self.bn1 = nn.BatchNorm2d(mid_channels)
#         self.conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False) # Process reduced dimension
#         self.bn2 = nn.BatchNorm2d(mid_channels)
#         self.expand = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False) # Expand dimension
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#         self.downsample = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             # Adjust the residual path to have the same shape as the output if necessary
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         residual = self.downsample(x)

#         out = self.relu(self.bn1(self.reduce(x)))
#         out = self.relu(self.bn2(self.conv(out)))
#         out = self.bn3(self.expand(out))

#         out += residual
#         out = self.relu(out)

#         return out
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_channels=None, stride=1):
        super(Bottleneck, self).__init__()
        if internal_channels is None:
            internal_channels = out_channels // 4

        # The bottleneck still consists of three convolutions but now ensures the final output has 3 channels.
        self.reduce = nn.Conv2d(in_channels, internal_channels, kernel_size=1, stride=stride, bias=False)  # Reduce dimension
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.conv = nn.Conv2d(internal_channels, internal_channels, kernel_size=3, stride=1, padding=1, bias=False)  # Process reduced dimension
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.expand = nn.Conv2d(internal_channels, 3, kernel_size=1, bias=False)  # Expand dimension to 3 channels
        self.bn3 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != 3:
            # Adjust the residual path to have the same shape as the output if necessary
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(3)
            )

    def forward(self, x):
        residual = self.downsample(x)

        out = self.relu(self.bn1(self.reduce(x)))
        out = self.relu(self.bn2(self.conv(out)))
        out = self.bn3(self.expand(out))

        out += residual
        out = self.relu(out)

        return out


class Im2LatexModel(nn.Module):
    def __init__(self, num_decoder_layers=6, num_heads=8, hidden_dim=512, max_seq_length=512, height=7, width=7):
        super(Im2LatexModel, self).__init__()
        # Downscales and then processes through a bottleneck for efficient dimensionality reduction
        self.initial_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Bottleneck(in_channels=64, out_channels=256),  # Example bottleneck
        )

        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])
        self.proj = nn.Linear(512, hidden_dim)

        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_layer = nn.Linear(hidden_dim, max_seq_length)
        self.positional_encoding = PositionalEncoding2D(hidden_dim, height, width)

    def forward(self, x, tgt):
        print(f'x: {x.shape}')
        print(f'tgt: {tgt.shape}')


        x = self.initial_layers(x)  # Initial CNN layers
        cnn_features = self.resnet18(x)  # ResNet-18 features
        print(f'cnn_features 1: {cnn_features.shape}')
        cnn_features = cnn_features.view(cnn_features.size(0), 512, -1).permute(2, 0, 1)  # Reshape for linear projection
        print(f'cnn_features 2: {cnn_features.shape}')
        cnn_features = self.proj(cnn_features)  # Project features to desired dimensionality (hidden_dim)
        print(f'cnn_features 3: {cnn_features.shape}')

        # Apply 2D positional encoding correctly
        # Here, ensure the tensor is reshaped properly to match positional encoding dimensions.
        # cnn_features now have shape: [seq_len, batch, hidden_dim] from the projection
        # We need to reshape it to [batch, hidden_dim, height, width] format
        # Assume the resulting feature map size after ResNet and before projection is HxW
        H, W = 7, 7  # You might need to adjust these dimensions based on actual feature map size
        cnn_features = cnn_features.permute(1, 2, 0)  # Rearrange to [batch, hidden_dim, seq_len]
        print(f'cnn_features 4: {cnn_features.shape}')
        cnn_features = cnn_features.view(-1, 512, H, W)  # Reshape to [batch, hidden_dim, H, W]
        print(f'cnn_features 5: {cnn_features.shape}')
        cnn_features = self.positional_encoding(cnn_features)  # Apply positional encoding
        print(f'cnn_features 6: {cnn_features.shape}')
        cnn_features = cnn_features.view(cnn_features.size(0), 512, -1).permute(2, 0, 1)  # Reshape back for transformer decoder
        print(f'cnn_features 7: {cnn_features.shape}')

        output = self.transformer_decoder(tgt, cnn_features)  # Transformer decoder
        print(f'output 1: {output.shape}')
        output = self.output_layer(output)  # Final output layer
        print(f'output 2: {output.shape}')

        return output


