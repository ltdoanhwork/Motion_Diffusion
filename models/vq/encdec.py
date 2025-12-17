import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vq.resnet import Resnet1D

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # GroupNorm thường ổn định hơn BatchNorm trong các Attention Block
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        
        # Query, Key, Value mapping
        self.q = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
        # Output projection
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Computing Attention map
        # (Batch, Channels, Time) -> (Batch, Time, Channels)
        b, c, t = q.shape
        
        q = q.permute(0, 2, 1) # (b, t, c)
        k = k.reshape(b, c, t) # (b, c, t) - giữ nguyên để nhân ma trận
        
        # bmm: batch matrix multiplication
        w_ = torch.bmm(q, k) # (b, t, t)
        w_ = w_ * (int(c)**(-0.5)) # Scale factor
        w_ = F.softmax(w_, dim=2)

        # Attend to values
        w_ = w_.permute(0, 2, 1) 
        v = v.permute(0, 2, 1)  
        
        h_ = torch.bmm(w_, v) 

        h_ = h_.permute(0, 2, 1)
        
        h_ = self.proj_out(h_)

        return x + h_

class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 use_attention=True): # Thêm tham số use_attention
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        
        # --- ATTENTION INSERTION ---
        if use_attention:
            # Thêm Attention ở Bottleneck (độ phân giải thấp nhất)
            blocks.append(AttnBlock(width))
        # ---------------------------

        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 use_attention=True): # Thêm tham số use_attention
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        
        # --- ATTENTION INSERTION ---
        if use_attention:
            # Thêm Attention ngay sau khi chiếu từ Codebook -> Feature space
            blocks.append(AttnBlock(width))
        # ---------------------------

        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)