# import torch
# import torch.nn as nn
# from models.vq.encdec import Encoder, Decoder
# from models.vq.residual_vq import ResidualVQ

# class DiagonalGaussianDistribution(object):
#     def __init__(self, parameters, deterministic=False):
#         self.parameters = parameters
#         self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
#         self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
#         self.deterministic = deterministic
#         self.std = torch.exp(0.5 * self.logvar)
#         self.var = torch.exp(self.logvar)
#         if self.deterministic:
#             self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

#     def sample(self):
#         x = self.mean + self.std * torch.randn_like(self.mean).to(device=self.parameters.device)
#         return x

#     def kl(self, other=None):
#         if self.deterministic:
#             return torch.Tensor([0.]).to(device=self.parameters.device)
#         else:
#             if other is None:
#                 return 0.5 * torch.sum(torch.pow(self.mean, 2)
#                                        + self.var - 1.0 - self.logvar,
#                                        dim=[1, 2])
#             else:
#                 return 0.5 * torch.sum(
#                     torch.pow(self.mean - other.mean, 2) / other.var
#                     + self.var / other.var - 1.0 - self.logvar + other.logvar,
#                     dim=[1, 2])

#     def mode(self):
#         return self.mean

# class RVQVAE(nn.Module):
#     def __init__(self,
#                  args,
#                  input_width=263,
#                  nb_code=1024,
#                  code_dim=512,
#                  output_emb_width=512,
#                  down_t=3,
#                  stride_t=2,
#                  width=512,
#                  depth=3,
#                  dilation_growth_rate=3,
#                  activation='relu',
#                  norm=None,
#                  embed_dim=512,
#                  double_z=False,
#                  **kwargs):

#         super().__init__()
#         self.code_dim = code_dim
#         self.num_code = nb_code
#         self.embed_dim = embed_dim
#         self.double_z = double_z
#         self.input_width = input_width # Store input dimension for shape checking
        
#         # Ensure output_emb_width matches code_dim
#         assert output_emb_width == code_dim
        
#         self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
#                                dilation_growth_rate, activation=activation, norm=norm)
#         self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
#                                dilation_growth_rate, activation=activation, norm=norm)
        
#         rvqvae_config = {
#             'num_quantizers': args.num_quantizers,
#             'shared_codebook': args.shared_codebook,
#             'quantize_dropout_prob': args.quantize_dropout_prob,
#             'quantize_dropout_cutoff_index': 0,
#             'nb_code': nb_code,
#             'code_dim': code_dim, 
#             'args': args,
#         }
#         self.quantizer = ResidualVQ(**rvqvae_config)

#         if self.double_z:
#             self.quant_conv = nn.Conv1d(output_emb_width, 2 * embed_dim, 1)
#             self.post_quant_conv = nn.Conv1d(embed_dim, output_emb_width, 1)

#     def preprocess(self, x):
#         # (bs, T, D) -> (bs, D, T)
#         x = x.permute(0, 2, 1).float()
#         return x

#     def postprocess(self, x):
#         # (bs, D, T) ->  (bs, T, D)
#         return x.permute(0, 2, 1)

#     def encode(self, x):
#         # x: (N, T, D)
#         x_in = self.preprocess(x) # (N, D, T)
#         h = self.encoder(x_in)    # (N, C, T)
        
#         if self.double_z:
#             moments = self.quant_conv(h)
#             posterior = DiagonalGaussianDistribution(moments)
#             return posterior
#         else:
#             code_idx, all_codes = self.quantizer.quantize(h, return_latent=True)
#             return code_idx, all_codes

#     def decode(self, z):
#         # z: (N, C, T)
#         if self.double_z:
#             z = self.post_quant_conv(z)
            
#         x_out = self.decoder(z) 
        
#         # Robust Shape Check:
#         if x_out.shape[1] == self.input_width:
#             out = self.postprocess(x_out)
#         else:
#             out = x_out
            
#         return out

#     def quantize(self, z):
#         """
#         Used by the trainer in VQ-KL mode.
#         Pass explicit temperature=0.0 to avoid NoneType error.
#         Clone input z to avoid inplace operation errors during backward pass.
#         """
#         # z: (N, C, T)
#         # Clone z here because it is used in both decode() and quantize() branches.
#         # This prevents "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
#         x_quantized, code_idx, commit_loss, perplexity = self.quantizer(z.clone(), sample_codebook_temp=0.0)
#         return x_quantized, commit_loss, (None, None, code_idx)

#     def forward(self, x):
#         # Standard VQ-VAE forward pass
#         x_in = self.preprocess(x)
#         x_encoder = self.encoder(x_in)
        
#         x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
#         x_out = self.decoder(x_quantized)
        
#         # Same robust shape check as decode
#         if x_out.shape[1] == self.input_width:
#             out = self.postprocess(x_out)
#         else:
#             out = x_out
        
#         return out, commit_loss, perplexity

#     def forward_decoder(self, x):
#         x_d = self.quantizer.get_codes_from_indices(x)
#         x = torch.stack(x_d, dim=0).sum(dim=0)
#         x_out = self.decoder(x)
        
#         if x_out.shape[1] == self.input_width:
#             out = self.postprocess(x_out)
#         else:
#             out = x_out
            
#         return out

# class LengthEstimator(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(LengthEstimator, self).__init__()
#         nd = 512
#         self.output = nn.Sequential(
#             nn.Linear(input_size, nd),
#             nn.LayerNorm(nd),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Dropout(0.2),
#             nn.Linear(nd, nd // 2),
#             nn.LayerNorm(nd // 2),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Dropout(0.2),
#             nn.Linear(nd // 2, nd // 4),
#             nn.LayerNorm(nd // 4),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Linear(nd // 4, output_size)
#         )

#         self.output.apply(self.__init_weights)

#     def __init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def forward(self, text_emb):
#         return self.output(text_emb)

"""
Enhanced VQ-VAE with Self-Attention and Skip Connections
Better reconstruction quality for fine details (hands, fingers)
"""
import os
import sys

current_file_path = os.path.abspath(__file__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from models.vq.encdec import EncoderWithAttention, DecoderWithAttention
from models.vq.residual_vq import ResidualVQ


class DiagonalGaussianDistribution(object):
    """Diagonal Gaussian distribution for KL divergence"""
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn_like(
            self.mean, device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.], device=self.parameters.device)
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2]
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2]
                )

    def mode(self):
        return self.mean


class RVQVAEWithAttention(nn.Module):
    """
    Enhanced VQ-VAE with Self-Attention and Skip Connections
    
    Improvements:
    1. Self-Attention: Captures long-range temporal dependencies
    2. Skip Connections: Preserves fine details during reconstruction
    3. Cross-Attention: Sophisticated feature fusion (optional)
    
    Architecture:
        Encoder (with attention) 
        → VQ Quantization
        → KL Divergence (optional)
        → Decoder (with attention + skip connections)
    """
    def __init__(self,
                 args,
                 input_width=264,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 embed_dim=512,
                 double_z=False,
                 # Attention parameters
                 use_attention=True,
                 attention_type='self',  # 'self' or 'nonlocal'
                 num_attention_heads=8,
                 encoder_attention_layers=[2],  # Add attention at encoder layer 2 (bottleneck)
                 decoder_attention_layers=[1],  # Add attention at decoder layer 1
                 # Skip connection parameters
                 use_skip_connections=True,
                 skip_connection_type='concat',  # 'concat' or 'cross_attn'
                 **kwargs):
        """
        Args:
            use_attention: Enable self-attention in encoder/decoder
            attention_type: 'self' (multi-head) or 'nonlocal' (non-local block)
            num_attention_heads: Number of attention heads
            encoder_attention_layers: Which encoder layers get attention
            decoder_attention_layers: Which decoder layers get attention
            use_skip_connections: Enable U-Net style skip connections
            skip_connection_type: 
                - 'concat': Direct concatenation (simple, fast)
                - 'cross_attn': Cross-attention fusion (sophisticated, slower)
        """
        super().__init__()
        
        self.code_dim = code_dim
        self.num_code = nb_code
        self.embed_dim = embed_dim
        self.double_z = double_z
        self.input_width = input_width
        self.use_skip_connections = use_skip_connections
        
        assert output_emb_width == code_dim, "output_emb_width must equal code_dim"
        
        print("\n" + "="*70)
        print("INITIALIZING ENHANCED VQ-VAE WITH ATTENTION")
        print("="*70)
        
        # Enhanced Encoder with Attention
        self.encoder = EncoderWithAttention(
            input_emb_width=input_width,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            use_attention=use_attention,
            attention_type=attention_type,
            num_attention_heads=num_attention_heads,
            attention_layers=encoder_attention_layers,
            return_skips=use_skip_connections  # Return skip features for decoder
        )
        
        # Enhanced Decoder with Attention and Skip Connections
        self.decoder = DecoderWithAttention(
            input_emb_width=input_width,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
            use_attention=use_attention,
            attention_type=attention_type,
            num_attention_heads=num_attention_heads,
            attention_layers=decoder_attention_layers,
            use_skip_connections=use_skip_connections,
            skip_connection_type=skip_connection_type
        )
        
        # VQ Quantizer
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim': code_dim,
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)
        
        # KL divergence (optional)
        if self.double_z:
            self.quant_conv = nn.Conv1d(output_emb_width, 2 * embed_dim, 1)
            self.post_quant_conv = nn.Conv1d(embed_dim, output_emb_width, 1)
            print(f"[INFO] KL divergence enabled (double_z=True)")
        
        print("="*70 + "\n")

    def preprocess(self, x):
        """(B, T, D) -> (B, D, T)"""
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        """(B, D, T) -> (B, T, D)"""
        return x.permute(0, 2, 1)

    def encode(self, x):
        """
        Encode motion to latent space
        Args:
            x: (B, T, D) motion sequence
        Returns:
            If double_z=True: DiagonalGaussianDistribution
            Otherwise: (code_idx, all_codes)
        """
        x_in = self.preprocess(x)  # (B, D, T)
        
        # Encode (with skip features if enabled)
        if self.use_skip_connections:
            h, skip_features = self.encoder(x_in)
            # Store skip features for decoder
            self.skip_features = skip_features
        else:
            h = self.encoder(x_in)
            self.skip_features = None
        
        if self.double_z:
            # KL mode: return posterior distribution
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            return posterior
        else:
            # VQ mode: return quantized codes
            code_idx, all_codes = self.quantizer.quantize(h, return_latent=True)
            return code_idx, all_codes

    def decode(self, z):
        """
        Decode latent to motion
        Args:
            z: (B, C, T') latent tensor
        Returns:
            (B, T, D) reconstructed motion
        """
        if self.double_z:
            z = self.post_quant_conv(z)
        
        # Decode with skip connections
        if self.use_skip_connections and hasattr(self, 'skip_features'):
            x_out = self.decoder(z, skip_features=self.skip_features)
        else:
            x_out = self.decoder(z, skip_features=None)
        
        # x_out is already (B, T, D) from decoder
        return x_out

    def quantize(self, z):
        """
        Quantize latent codes (for VQ-KL mode)
        Args:
            z: (B, C, T') latent tensor
        Returns:
            (x_quantized, commit_loss, perplexity_info)
        """
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(
            z.clone(), 
            sample_codebook_temp=0.0
        )
        return x_quantized, commit_loss, (None, None, code_idx)

    def forward(self, x):
        """
        Standard VQ-VAE forward pass
        Args:
            x: (B, T, D) motion sequence
        Returns:
            (reconstructed_motion, commit_loss, perplexity)
        """
        x_in = self.preprocess(x)
        
        # Encode
        if self.use_skip_connections:
            x_encoder, skip_features = self.encoder(x_in)
            self.skip_features = skip_features
        else:
            x_encoder = self.encoder(x_in)
            self.skip_features = None
        
        # Quantize
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(
            x_encoder, 
            sample_codebook_temp=0.5
        )
        
        # Decode
        if self.use_skip_connections:
            x_out = self.decoder(x_quantized, skip_features=self.skip_features)
        else:
            x_out = self.decoder(x_quantized, skip_features=None)
        
        return x_out, commit_loss, perplexity

    def forward_decoder(self, x):
        """
        Decode from code indices
        Args:
            x: Code indices
        Returns:
            (B, T, D) reconstructed motion
        """
        x_d = self.quantizer.get_codes_from_indices(x)
        x = torch.stack(x_d, dim=0).sum(dim=0)
        
        # Note: Skip connections not available when decoding from indices
        x_out = self.decoder(x, skip_features=None)
        return x_out


# Backward compatibility alias
RVQVAE = RVQVAEWithAttention


class LengthEstimator(nn.Module):
    """Motion length estimator (unchanged)"""
    def __init__(self, input_size, output_size):
        super().__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nd // 4, output_size)
        )
        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)


# Test
if __name__ == "__main__":
    import argparse
    
    print("\nTesting Enhanced VQ-VAE with Attention and Skip Connections")
    print("="*70)
    
    # Mock args
    args = argparse.Namespace(
        num_quantizers=10,
        shared_codebook=False,
        quantize_dropout_prob=0.0,
        mu=0.99
    )
    
    # Test 1: With attention and concat skip connections
    print("\n[Test 1] With Self-Attention + Concat Skip Connections")
    model = RVQVAEWithAttention(
        args=args,
        input_width=264,
        nb_code=1024,
        code_dim=512,
        output_emb_width=512,
        down_t=3,
        width=512,
        depth=3,
        double_z=True,
        use_attention=True,
        attention_type='self',
        num_attention_heads=8,
        encoder_attention_layers=[2],
        decoder_attention_layers=[1],
        use_skip_connections=True,
        skip_connection_type='concat'
    )
    
    x = torch.randn(2, 360, 264)
    out, commit_loss, perplexity = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Commit loss: {commit_loss.item():.4f}")
    
    # Test 2: With cross-attention skip connections
    print("\n[Test 2] With Self-Attention + Cross-Attention Skip Connections")
    model_cross = RVQVAEWithAttention(
        args=args,
        input_width=264,
        nb_code=1024,
        code_dim=512,
        output_emb_width=512,
        down_t=3,
        width=512,
        depth=3,
        double_z=True,
        use_attention=True,
        use_skip_connections=True,
        skip_connection_type='cross_attn'
    )
    
    out, commit_loss, perplexity = model_cross(x)
    print(f"Output: {out.shape}")
    
    # Test 3: Encode-decode cycle
    print("\n[Test 3] Encode-Decode Cycle")
    posterior = model.encode(x)
    z = posterior.sample()
    print(f"Latent shape: {z.shape}")
    
    reconstructed = model.decode(z)
    print(f"Reconstructed: {reconstructed.shape}")
    
    print("\n" + "="*70)
    print("All tests passed! ✓")