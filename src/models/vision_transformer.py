"""
Vision Transformer implementation for low-light image enhancement.
Uses patch-based processing for efficient computation.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        assert self.head_dim * n_heads == embed_dim, "embed_dim must be divisible by n_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        output = self.out_proj(attended)
        return output


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for image enhancement.
    
    Args:
        img_size (int): Input image size
        patch_size (int): Patch size for tokenization
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        embed_dim (int): Embedding dimension
        n_layers (int): Number of transformer layers
        n_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        out_channels=3,
        embed_dim=768,
        n_layers=12,
        n_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Reconstruction head
        self.head = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Reconstruction
        x = self.head(x)  # (B, n_patches, patch_size^2 * out_channels)
        
        # Reshape to image
        x = x.view(
            batch_size,
            int(self.img_size // self.patch_size),
            int(self.img_size // self.patch_size),
            self.patch_size,
            self.patch_size,
            -1
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, -1, self.img_size, self.img_size)
        
        return torch.sigmoid(x)
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightViT(VisionTransformer):
    """Lightweight Vision Transformer for mobile deployment."""
    
    def __init__(self, img_size=224, **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=16,
            embed_dim=384,
            n_layers=6,
            n_heads=6,
            mlp_ratio=2.0,
            **kwargs
        )


def create_vit_model(model_type='standard', img_size=224, **kwargs):
    """
    Factory function to create different ViT variants.
    
    Args:
        model_type (str): Type of model ('standard', 'lightweight')
        img_size (int): Input image size
        **kwargs: Additional arguments for model initialization
    
    Returns:
        torch.nn.Module: ViT model instance
    """
    if model_type == 'standard':
        return VisionTransformer(img_size=img_size, **kwargs)
    elif model_type == 'lightweight':
        return LightweightViT(img_size=img_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = create_vit_model('lightweight', img_size=224)
    model = model.to(device)
    
    # Test input
    x = torch.randn(2, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(x)
        
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Device: {device}")
