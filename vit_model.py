#!/usr/bin/env python3
"""
Vision Transformer for Low-Light Enhancement
Clean implementation for comparison with U-Net
"""

import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)        # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (B, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer for image enhancement"""
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(0.1)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder for image reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Decode patches back to image
        x = self.decoder(x)  # (B, n_patches, patch_size^2 * channels)
        
        # Reshape to image
        x = x.reshape(B, self.n_patches, self.patch_size, self.patch_size, 3)
        
        # Rearrange patches to form image
        patches_per_row = self.img_size // self.patch_size
        x = x.reshape(B, patches_per_row, patches_per_row, self.patch_size, self.patch_size, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, 3, patches_per_row, patch_size, patches_per_row, patch_size)
        x = x.reshape(B, 3, self.img_size, self.img_size)
        
        return x

def create_vit(device='cpu'):
    """Create and return Vision Transformer model"""
    model = VisionTransformer(
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=384,  # Smaller than standard for efficiency
        depth=6,        # Fewer layers for speed
        num_heads=6
    )
    model = model.to(device)
    return model

if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_vit(device)
    test_input = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print("Vision Transformer created successfully!")
