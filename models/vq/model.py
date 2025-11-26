import torch
import torch.nn as nn
from models.vq.encdec import Encoder, Decoder
from models.vq.residual_vq import ResidualVQ

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.]).to(device=self.parameters.device)
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2])

    def mode(self):
        return self.mean

class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
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
                 **kwargs):

        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.embed_dim = embed_dim
        self.double_z = double_z
        self.input_width = input_width # Store input dimension for shape checking
        
        # Ensure output_emb_width matches code_dim
        assert output_emb_width == code_dim
        
        self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        
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

        if self.double_z:
            self.quant_conv = nn.Conv1d(output_emb_width, 2 * embed_dim, 1)
            self.post_quant_conv = nn.Conv1d(embed_dim, output_emb_width, 1)

    def preprocess(self, x):
        # (bs, T, D) -> (bs, D, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, D, T) ->  (bs, T, D)
        return x.permute(0, 2, 1)

    def encode(self, x):
        # x: (N, T, D)
        x_in = self.preprocess(x) # (N, D, T)
        h = self.encoder(x_in)    # (N, C, T)
        
        if self.double_z:
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            return posterior
        else:
            code_idx, all_codes = self.quantizer.quantize(h, return_latent=True)
            return code_idx, all_codes

    def decode(self, z):
        # z: (N, C, T)
        if self.double_z:
            z = self.post_quant_conv(z)
            
        x_out = self.decoder(z) 
        
        # Robust Shape Check:
        if x_out.shape[1] == self.input_width:
            out = self.postprocess(x_out)
        else:
            out = x_out
            
        return out

    def quantize(self, z):
        """
        Used by the trainer in VQ-KL mode.
        Pass explicit temperature=0.0 to avoid NoneType error.
        Clone input z to avoid inplace operation errors during backward pass.
        """
        # z: (N, C, T)
        # Clone z here because it is used in both decode() and quantize() branches.
        # This prevents "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(z.clone(), sample_codebook_temp=0.0)
        return x_quantized, commit_loss, (None, None, code_idx)

    def forward(self, x):
        # Standard VQ-VAE forward pass
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        x_out = self.decoder(x_quantized)
        
        # Same robust shape check as decode
        if x_out.shape[1] == self.input_width:
            out = self.postprocess(x_out)
        else:
            out = x_out
        
        return out, commit_loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        x = torch.stack(x_d, dim=0).sum(dim=0)
        x_out = self.decoder(x)
        
        if x_out.shape[1] == self.input_width:
            out = self.postprocess(x_out)
        else:
            out = x_out
            
        return out

class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
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