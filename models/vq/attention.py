"""
Self-Attention modules for motion VQ-VAE
Helps capture long-range temporal dependencies and global context
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention1D(nn.Module):
    """
    Self-Attention for 1D temporal sequences
    Allows each timestep to attend to all other timesteps
    """
    def __init__(self, channels, num_heads=8, dropout=0.0):
        """
        Args:
            channels: Number of input channels
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        # Multi-head attention components
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, T) - input tensor
        Returns:
            (B, C, T) - output tensor with same shape
        """
        B, C, T = x.shape
        h = self.norm(x)
        
        # Generate Q, K, V
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        # (B, C, T) -> (B, num_heads, C/num_heads, T)
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, T)
        k = k.view(B, self.num_heads, head_dim, T)
        v = v.view(B, self.num_heads, head_dim, T)
        
        # Scaled dot-product attention
        # Q * K^T / sqrt(d_k)
        scale = head_dim ** -0.5
        attn = torch.einsum('bhdt,bhds->bhts', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Attention * V
        out = torch.einsum('bhts,bhds->bhdt', attn, v)
        
        # Reshape back
        out = out.reshape(B, C, T)
        out = self.proj_out(out)
        
        # Residual connection
        return x + out


class NonLocalBlock1D(nn.Module):
    """
    Non-Local Block for capturing global context
    More efficient than full self-attention for long sequences
    """
    def __init__(self, channels, sub_sample=True, bn_layer=True):
        """
        Args:
            channels: Number of input channels
            sub_sample: Use downsampling to reduce computation
            bn_layer: Use batch normalization
        """
        super().__init__()
        self.sub_sample = sub_sample
        
        # Compute intermediate channels
        inter_channels = channels // 2
        if inter_channels == 0:
            inter_channels = 1
        
        # Theta, Phi, G paths
        self.theta = nn.Conv1d(channels, inter_channels, 1)
        self.phi = nn.Conv1d(channels, inter_channels, 1)
        self.g = nn.Conv1d(channels, inter_channels, 1)
        
        # Output projection
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv1d(inter_channels, channels, 1),
                nn.BatchNorm1d(channels)
            )
            # Initialize to zero (residual path starts as identity)
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv1d(inter_channels, channels, 1)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        
        # Downsampling for efficiency
        if sub_sample:
            self.g = nn.Sequential(
                self.g,
                nn.MaxPool1d(kernel_size=2)
            )
            self.phi = nn.Sequential(
                self.phi,
                nn.MaxPool1d(kernel_size=2)
            )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T)
        """
        batch_size = x.size(0)
        
        # Generate theta, phi, g
        g_x = self.g(x).view(batch_size, -1, x.size(2) // 2 if self.sub_sample else x.size(2))
        theta_x = self.theta(x).view(batch_size, -1, x.size(2))
        phi_x = self.phi(x).view(batch_size, -1, x.size(2) // 2 if self.sub_sample else x.size(2))
        
        # Compute attention
        # (B, C', T) x (B, C', T') -> (B, T, T')
        f = torch.matmul(theta_x.transpose(1, 2), phi_x)
        f_div_C = F.softmax(f, dim=-1)
        
        # Apply attention to g
        # (B, T, T') x (B, C', T') -> (B, T, C')
        y = torch.matmul(f_div_C, g_x.transpose(1, 2))
        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, -1, x.size(2))
        
        # Output projection
        W_y = self.W(y)
        
        # Residual connection
        return x + W_y


class CrossAttention1D(nn.Module):
    """
    Cross-Attention between encoder and decoder features (for skip connections)
    """
    def __init__(self, query_dim, key_dim, num_heads=8, dropout=0.0):
        """
        Args:
            query_dim: Channels of query (decoder features)
            key_dim: Channels of key/value (encoder features)
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert query_dim % num_heads == 0
        
        self.to_q = nn.Conv1d(query_dim, query_dim, 1)
        self.to_k = nn.Conv1d(key_dim, query_dim, 1)
        self.to_v = nn.Conv1d(key_dim, query_dim, 1)
        
        self.proj_out = nn.Conv1d(query_dim, query_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
        self.norm_q = nn.GroupNorm(8, query_dim)
        self.norm_kv = nn.GroupNorm(8, key_dim)
        
    def forward(self, query, key_value):
        """
        Args:
            query: (B, C_q, T_q) - decoder features
            key_value: (B, C_k, T_k) - encoder features
        Returns:
            (B, C_q, T_q)
        """
        B, C_q, T_q = query.shape
        _, C_k, T_k = key_value.shape
        
        # Normalize
        query = self.norm_q(query)
        key_value = self.norm_kv(key_value)
        
        # Generate Q, K, V
        q = self.to_q(query)
        k = self.to_k(key_value)
        v = self.to_v(key_value)
        
        # Reshape for multi-head
        q = q.view(B, self.num_heads, self.head_dim, T_q)
        k = k.view(B, self.num_heads, self.head_dim, T_k)
        v = v.view(B, self.num_heads, self.head_dim, T_k)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdt,bhds->bhts', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.einsum('bhts,bhds->bhdt', attn, v)
        out = out.reshape(B, C_q, T_q)
        out = self.proj_out(out)
        
        return out


class TemporalAttentionBlock(nn.Module):
    """
    Complete attention block with pre-norm and residual connection
    Can be inserted into ResNet-based architectures
    """
    def __init__(self, channels, num_heads=8, use_nonlocal=False, dropout=0.0):
        """
        Args:
            channels: Number of channels
            num_heads: Number of attention heads
            use_nonlocal: Use NonLocal block instead of Self-Attention
            dropout: Dropout rate
        """
        super().__init__()
        
        if use_nonlocal:
            self.attn = NonLocalBlock1D(channels, sub_sample=True, bn_layer=True)
        else:
            self.attn = SelfAttention1D(channels, num_heads, dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.Conv1d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 4, channels, 1),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T)
        """
        # Attention with residual
        x = self.attn(x)
        
        # Feed-forward with residual
        x = x + self.ff(x)
        
        return x


# Example usage in encoder/decoder
if __name__ == "__main__":
    # Test SelfAttention1D
    x = torch.randn(4, 256, 64)  # (batch, channels, time)
    attn = SelfAttention1D(256, num_heads=8)
    out = attn(x)
    print(f"SelfAttention1D: {x.shape} -> {out.shape}")
    
    # Test NonLocalBlock1D
    nonlocal_block = NonLocalBlock1D(256, sub_sample=True)
    out = nonlocal_block(x)
    print(f"NonLocalBlock1D: {x.shape} -> {out.shape}")
    
    # Test TemporalAttentionBlock
    attn_block = TemporalAttentionBlock(256, num_heads=8)
    out = attn_block(x)
    print(f"TemporalAttentionBlock: {x.shape} -> {out.shape}")
    
    # Test CrossAttention1D
    query = torch.randn(4, 256, 32)
    key_value = torch.randn(4, 512, 64)
    cross_attn = CrossAttention1D(query_dim=256, key_dim=512, num_heads=8)
    out = cross_attn(query, key_value)
    print(f"CrossAttention1D: query {query.shape}, kv {key_value.shape} -> {out.shape}")