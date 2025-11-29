# import os
# import sys
# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

# import torch.nn as nn
# from models.vq.resnet import Resnet1D


# class Encoder(nn.Module):
#     def __init__(self,
#                  input_emb_width=3,
#                  output_emb_width=512,
#                  down_t=2,
#                  stride_t=2,
#                  width=512,
#                  depth=3,
#                  dilation_growth_rate=3,
#                  activation='relu',
#                  norm=None):
#         super().__init__()

#         blocks = []
#         filter_t, pad_t = stride_t * 2, stride_t // 2
#         blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())

#         for i in range(down_t):
#             input_dim = width
#             block = nn.Sequential(
#                 nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
#                 Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
#             )
#             blocks.append(block)
#         blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
#         self.model = nn.Sequential(*blocks)

#     def forward(self, x):
#         return self.model(x)


# class Decoder(nn.Module):
#     def __init__(self,
#                  input_emb_width=3,
#                  output_emb_width=512,
#                  down_t=2,
#                  stride_t=2,
#                  width=512,
#                  depth=3,
#                  dilation_growth_rate=3,
#                  activation='relu',
#                  norm=None):
#         super().__init__()
#         blocks = []

#         blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())
#         for i in range(down_t):
#             out_dim = width
#             block = nn.Sequential(
#                 Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
#                 nn.Upsample(scale_factor=2, mode='nearest'),
#                 nn.Conv1d(width, out_dim, 3, 1, 1)
#             )
#             blocks.append(block)
#         blocks.append(nn.Conv1d(width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())
#         blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
#         self.model = nn.Sequential(*blocks)

#     def forward(self, x):
#         x = self.model(x)
#         return x.permute(0, 2, 1)

"""
Enhanced Encoder-Decoder with Self-Attention and Skip Connections
Improves reconstruction quality, especially for fine details (hands, fingers)
"""
import os
import sys

current_file_path = os.path.abspath(__file__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from models.vq.resnet import Resnet1D
from models.vq.attention import (
    SelfAttention1D, 
    NonLocalBlock1D, 
    TemporalAttentionBlock,
    CrossAttention1D
)


class EncoderWithAttention(nn.Module):
    """
    Enhanced Encoder with Self-Attention
    Architecture:
    - Initial Conv1d
    - Down-sampling blocks with ResNet
    - Self-Attention at bottleneck (lowest resolution)
    - Final Conv1d projection
    """
    def __init__(self,
                 input_emb_width=264,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 use_attention=True,
                 attention_type='self',  # 'self' or 'nonlocal'
                 num_attention_heads=8,
                 attention_layers=[2],  # Which down-sampling stages to add attention
                 return_skips=True):    # Return intermediate features for skip connections
        """
        Args:
            input_emb_width: Input dimension (e.g., 264 for BEAT)
            output_emb_width: Output latent dimension
            down_t: Number of down-sampling stages
            stride_t: Stride for down-sampling
            width: Base channel width
            depth: Depth of ResNet blocks
            dilation_growth_rate: Dilation growth rate in ResNet
            activation: Activation function
            norm: Normalization type
            use_attention: Enable self-attention
            attention_type: 'self' or 'nonlocal'
            num_attention_heads: Number of attention heads
            attention_layers: List of stages to add attention (0-indexed)
            return_skips: Return skip connections for U-Net style decoder
        """
        super().__init__()
        
        self.return_skips = return_skips
        self.down_t = down_t
        
        # Initial convolution
        self.conv_in = nn.Sequential(
            nn.Conv1d(input_emb_width, width, 3, 1, 1),
            nn.ReLU()
        )
        
        # Down-sampling blocks with optional attention
        self.down_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        filter_t = stride_t * 2
        pad_t = stride_t // 2
        
        for i in range(down_t):
            # Down-sampling + ResNet
            block = nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, 
                        activation=activation, norm=norm),
            )
            self.down_blocks.append(block)
            
            # Add attention at specified layers
            if use_attention and i in attention_layers:
                if attention_type == 'nonlocal':
                    attn = NonLocalBlock1D(width, sub_sample=True, bn_layer=True)
                else:
                    attn = TemporalAttentionBlock(
                        width, 
                        num_heads=num_attention_heads,
                        use_nonlocal=False,
                        dropout=0.1
                    )
                self.attention_blocks.append(attn)
                print(f"[INFO] Added {attention_type} attention at encoder layer {i}")
            else:
                self.attention_blocks.append(nn.Identity())
        
        # Final projection to latent space
        self.conv_out = nn.Conv1d(width, output_emb_width, 3, 1, 1)
        
        print(f"[INFO] EncoderWithAttention initialized:")
        print(f"  - Input: {input_emb_width} dims")
        print(f"  - Output: {output_emb_width} dims")
        print(f"  - Down-sampling stages: {down_t}")
        print(f"  - Attention: {attention_type if use_attention else 'None'}")
        print(f"  - Skip connections: {return_skips}")

    def forward(self, x):
        """
        Args:
            x: (B, C_in, T) input tensor
        Returns:
            If return_skips=True: (latent, skip_features)
            Otherwise: latent
        """
        skip_features = []
        
        # Initial conv
        h = self.conv_in(x)
        if self.return_skips:
            skip_features.append(h)
        
        # Down-sampling with attention
        for i, (down_block, attn_block) in enumerate(
            zip(self.down_blocks, self.attention_blocks)
        ):
            h = down_block(h)
            h = attn_block(h)  # Apply attention (or Identity if not used)
            
            if self.return_skips:
                skip_features.append(h)
        
        # Final projection
        h = self.conv_out(h)
        
        if self.return_skips:
            return h, skip_features
        else:
            return h


class DecoderWithAttention(nn.Module):
    """
    Enhanced Decoder with Self-Attention and Skip Connections
    Architecture:
    - Initial Conv1d
    - Up-sampling blocks with ResNet
    - Cross-Attention with encoder features (skip connections)
    - Self-Attention at bottleneck
    - Final Conv1d projection
    """
    def __init__(self,
                 input_emb_width=264,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 use_attention=True,
                 attention_type='self',
                 num_attention_heads=8,
                 attention_layers=[1],  # Which up-sampling stages to add attention
                 use_skip_connections=True,
                 skip_connection_type='concat'):  # 'concat' or 'cross_attn'
        """
        Args:
            use_skip_connections: Enable U-Net style skip connections
            skip_connection_type: 
                - 'concat': Direct concatenation (U-Net style)
                - 'cross_attn': Cross-attention fusion (more sophisticated)
        """
        super().__init__()
        
        self.use_skip_connections = use_skip_connections
        self.skip_connection_type = skip_connection_type
        self.down_t = down_t
        
        # Initial projection
        self.conv_in = nn.Sequential(
            nn.Conv1d(output_emb_width, width, 3, 1, 1),
            nn.ReLU()
        )
        
        # Up-sampling blocks with optional attention and skip connections
        self.up_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.skip_fusion_blocks = nn.ModuleList()
        
        for i in range(down_t):
            # ResNet + Up-sampling
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, 
                        reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, width, 3, 1, 1)
            )
            self.up_blocks.append(block)
            
            # Skip connection fusion
            if use_skip_connections:
                if skip_connection_type == 'concat':
                    # Concatenate and project back to width
                    fusion = nn.Sequential(
                        nn.Conv1d(width * 2, width, 1),
                        nn.ReLU()
                    )
                elif skip_connection_type == 'cross_attn':
                    # Cross-attention between decoder and encoder features
                    fusion = CrossAttention1D(
                        query_dim=width,
                        key_dim=width,
                        num_heads=num_attention_heads,
                        dropout=0.1
                    )
                else:
                    raise ValueError(f"Unknown skip_connection_type: {skip_connection_type}")
                
                self.skip_fusion_blocks.append(fusion)
            else:
                self.skip_fusion_blocks.append(nn.Identity())
            
            # Self-attention
            if use_attention and i in attention_layers:
                if attention_type == 'nonlocal':
                    attn = NonLocalBlock1D(width, sub_sample=False, bn_layer=True)
                else:
                    attn = TemporalAttentionBlock(
                        width,
                        num_heads=num_attention_heads,
                        use_nonlocal=False,
                        dropout=0.1
                    )
                self.attention_blocks.append(attn)
                print(f"[INFO] Added {attention_type} attention at decoder layer {i}")
            else:
                self.attention_blocks.append(nn.Identity())
        
        # Final layers
        self.conv_out = nn.Sequential(
            nn.Conv1d(width, width, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(width, input_emb_width, 3, 1, 1)
        )
        
        print(f"[INFO] DecoderWithAttention initialized:")
        print(f"  - Input: {output_emb_width} dims")
        print(f"  - Output: {input_emb_width} dims")
        print(f"  - Up-sampling stages: {down_t}")
        print(f"  - Attention: {attention_type if use_attention else 'None'}")
        print(f"  - Skip connections: {use_skip_connections} ({skip_connection_type})")

    def forward(self, x, skip_features=None):
        """
        Args:
            x: (B, C_latent, T') latent tensor
            skip_features: List of skip connection features from encoder
                           (if use_skip_connections=True)
        Returns:
            (B, T, C_out) output motion
        """
        # Initial projection
        h = self.conv_in(x)
        
        # Process skip features if provided
        if self.use_skip_connections and skip_features is not None:
            # Reverse skip features (decoder processes from low to high resolution)
            skip_features = skip_features[::-1]
            # Remove the first skip (from conv_in) as we don't use it
            skip_features = skip_features[1:]
        
        # Up-sampling with attention and skip connections
        for i, (up_block, attn_block, skip_fusion) in enumerate(
            zip(self.up_blocks, self.attention_blocks, self.skip_fusion_blocks)
        ):
            # Up-sample
            h = up_block(h)
            
            # Fuse with skip connection
            if self.use_skip_connections and skip_features is not None and i < len(skip_features):
                skip = skip_features[i]
                
                # Match temporal dimension (in case of mismatch due to rounding)
                if h.shape[2] != skip.shape[2]:
                    if h.shape[2] > skip.shape[2]:
                        h = h[:, :, :skip.shape[2]]
                    else:
                        skip = skip[:, :, :h.shape[2]]
                
                # Fuse
                if self.skip_connection_type == 'concat':
                    h = torch.cat([h, skip], dim=1)
                    h = skip_fusion(h)
                elif self.skip_connection_type == 'cross_attn':
                    # Cross-attention: query=decoder, key/value=encoder
                    h = h + skip_fusion(h, skip)
            
            # Self-attention
            h = attn_block(h)
        
        # Final projection
        x_out = self.conv_out(h)
        
        # (B, C, T) -> (B, T, C)
        return x_out.permute(0, 2, 1)


# Backward compatibility: original Encoder/Decoder without attention
class Encoder(nn.Module):
    """Original Encoder without attention (for backward compatibility)"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Remove attention-related kwargs
        kwargs.pop('use_attention', None)
        kwargs.pop('attention_type', None)
        kwargs.pop('num_attention_heads', None)
        kwargs.pop('attention_layers', None)
        kwargs.pop('return_skips', None)
        
        # Create encoder without attention
        self.encoder = EncoderWithAttention(
            *args, 
            use_attention=False, 
            return_skips=False,
            **kwargs
        )
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """Original Decoder without attention (for backward compatibility)"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Remove attention-related kwargs
        kwargs.pop('use_attention', None)
        kwargs.pop('attention_type', None)
        kwargs.pop('num_attention_heads', None)
        kwargs.pop('attention_layers', None)
        kwargs.pop('use_skip_connections', None)
        kwargs.pop('skip_connection_type', None)
        
        # Create decoder without attention or skip connections
        self.decoder = DecoderWithAttention(
            *args,
            use_attention=False,
            use_skip_connections=False,
            **kwargs
        )
    
    def forward(self, x):
        return self.decoder(x, skip_features=None)


# Test
if __name__ == "__main__":
    print("Testing Enhanced Encoder-Decoder with Attention and Skip Connections")
    print("="*70)
    
    # Test 1: Encoder with attention
    print("\n[Test 1] EncoderWithAttention")
    encoder = EncoderWithAttention(
        input_emb_width=264,
        output_emb_width=512,
        down_t=3,
        width=512,
        depth=3,
        use_attention=True,
        attention_type='self',
        attention_layers=[2],
        return_skips=True
    )
    
    x = torch.randn(4, 264, 360)  # (batch, channels, time)
    latent, skips = encoder(x)
    print(f"Input: {x.shape}")
    print(f"Latent: {latent.shape}")
    print(f"Skip features: {[s.shape for s in skips]}")
    
    # Test 2: Decoder with attention and skip connections (concat)
    print("\n[Test 2] DecoderWithAttention (concat)")
    decoder_concat = DecoderWithAttention(
        input_emb_width=264,
        output_emb_width=512,
        down_t=3,
        width=512,
        depth=3,
        use_attention=True,
        attention_type='self',
        attention_layers=[1],
        use_skip_connections=True,
        skip_connection_type='concat'
    )
    
    out = decoder_concat(latent, skip_features=skips)
    print(f"Output: {out.shape}")
    
    # Test 3: Decoder with cross-attention skip connections
    print("\n[Test 3] DecoderWithAttention (cross_attn)")
    decoder_cross = DecoderWithAttention(
        input_emb_width=264,
        output_emb_width=512,
        down_t=3,
        width=512,
        depth=3,
        use_attention=True,
        attention_type='self',
        attention_layers=[1],
        use_skip_connections=True,
        skip_connection_type='cross_attn'
    )
    
    out = decoder_cross(latent, skip_features=skips)
    print(f"Output: {out.shape}")
    
    print("\n" + "="*70)
    print("All tests passed! ✓")