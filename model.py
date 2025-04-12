import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic Convolutional Block: Conv -> ReLU -> Conv -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=3, features=[64, 128, 256, 512], action_space_size=14, action_embedding_dim=3):
        super().__init__()
        self.action_embedding = nn.Embedding(action_space_size, action_embedding_dim)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Encoder: Downsampling path
        prev_channels = in_channels + 4*action_embedding_dim
        for feature in features:
            self.encoder.append(ConvBlock(prev_channels, feature))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder: Upsampling path
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(ConvBlock(feature * 2, feature))

        # Final output layer (maps to 3-channel frame)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, actions, only_bottleneck=False):
        action_emb = self.action_embedding(actions)
        action_emb = action_emb.view((x.shape[0], -1)).unsqueeze(2).unsqueeze(3).repeat(1, 1, 210, 160)
        x = torch.concat([x, action_emb], dim=1)

        skip_connections = []

        # Encoder (downsampling)
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        if only_bottleneck:
            return x

        # Decoder (upsampling)
        skip_connections = skip_connections[::-1]  # Reverse for skip connections
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Up-convolution
            skip = skip_connections[i // 2]

            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat((skip, x), dim=1)  # Concatenate skip connection
            x = self.decoder[i + 1](x)  # Apply ConvBlock

        return self.final_conv(x)  # Output 3-channel frame
    

class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.fcs = nn.ModuleList()  # Store layers in a list

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.fcs.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim  # Update input size for next layer
        
        self.fcs.append(nn.Linear(prev_dim, output_dim))  # Final output layer

    def forward(self, x):
        for layer in self.fcs[:-1]:  # Apply activation to all but last
            x = F.relu(layer(x))
        return self.fcs[-1](x)  # No activation on output