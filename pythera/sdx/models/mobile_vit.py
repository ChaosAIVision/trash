import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, MobileViTModel
from datasets import load_dataset

class FourierEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, inputs):
        # inputs: (B,)
        if inputs.dim() > 1:
            inputs = inputs.squeeze(-1)
        freqs = torch.linspace(1.0, 10.0, self.embedding_dim // 2, device=inputs.device)
        inputs = inputs.unsqueeze(1)  # (B, 1)
        embeddings = torch.cat([torch.sin(inputs * freqs), torch.cos(inputs * freqs)], dim=1)
        return embeddings

class FourierEmbeddingAttention(nn.Module):
    def __init__(self, embedding_dim, in_channels):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(embedding_dim, in_channels)

    def forward(self, x, embedding):
        # Adjust feature map using Fourier Embedding
        attention = torch.sigmoid(self.fc(embedding))  # (B, C)
        return x * attention.unsqueeze(-1).unsqueeze(-1)

class ImprovedConvNeXtBlockWithFourierEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim=640):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.norm = nn.BatchNorm2d(in_channels)  # Changed to BatchNorm2d
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)
        self.silu = nn.SiLU()
        self.pwconv2 = nn.Linear(4 * in_channels, out_channels)
        self.transconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),  # Changed to SiLU activation
            nn.BatchNorm2d(out_channels)  # BatchNorm2d for better training stability
        )

        # Fourier Embedding Attention
        self.time_attention = FourierEmbeddingAttention(embedding_dim, in_channels)
        self.angle_attention = FourierEmbeddingAttention(embedding_dim, in_channels)

        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x, timesteps, angles):
        device = x.device  # Ensure everything is on the same device
        # Get Fourier embeddings
        time_embedding = FourierEmbedding(self.time_attention.embedding_dim)(timesteps).to(device).unsqueeze(-1).unsqueeze(-1).to(torch.bfloat16)
        angle_embedding = FourierEmbedding(self.angle_attention.embedding_dim)(angles).to(device).unsqueeze(-1).unsqueeze(-1).to(torch.bfloat16)

        # Adjust feature map with Fourier Embedding
        x = self.time_attention(x, time_embedding.squeeze(-1).squeeze(-1))
        x = self.angle_attention(x, angle_embedding.squeeze(-1).squeeze(-1))

        # Standard ConvNeXt operations
        x = self.dwconv(x)
        x = self.norm(x)  # BatchNorm2d applied
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.silu(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        # Global Pooling adjustment
        # pooled = self.global_pool(x).squeeze(-1).squeeze(-1)
        # pooled = self.fc(pooled).unsqueeze(-1).unsqueeze(-1)
        # x = x + pooled

        # Increase spatial resolution
        x = self.transconv(x)
        return x

class CustomMobileViT(nn.Module):
    def __init__(self, embedding_dim=640):
        super().__init__()
        self.base_model = MobileViTModel.from_pretrained(
            "apple/mobilevit-small",
            output_stride=8,
            neck_hidden_sizes=[
                16,
                32,
                64,
                96,
                128,
                160,
                640,  # Ensure output channels are 640
            ],
        )



         # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Improved ConvNeXt block
        self.convnext_block = ImprovedConvNeXtBlockWithFourierEmbedding(in_channels=640, out_channels=320, embedding_dim=embedding_dim)

    def forward(self, inputs, timesteps, angles):
        inputs = inputs.to(torch.bfloat16)
        timesteps = timesteps.to(torch.bfloat16)
        angles = angles.to(torch.bfloat16)
        device = next(self.parameters()).device  # Ensure device consistency
        timesteps = timesteps.to(device)
        angles = angles.to(device)

        outputs = self.base_model(**{k: v.to(device) for k, v in inputs.items()})
        last_hidden_state = outputs.last_hidden_state  # Shape: [B, 640, 32, 32]
        enhanced_output = self.convnext_block(last_hidden_state, timesteps, angles)  # Shape: [B, 320, 64, 64]
        return enhanced_output



