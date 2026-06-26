"""MultiPatchFormer — Multi-scale patch embedding with attention.

Pipeline: "large". Uses internal normalization (series stationarization).
Deps: einops, shared_layers.SelfAttention_Family.
"""

import math
import warnings
import torch
import torch.nn as nn
from einops import rearrange
from .base import BaseModel
from .shared_layers.SelfAttention_Family import AttentionLayer, FullAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden=512):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear_1(x); x = self.activation(x); x = self.linear_2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, mha, d_hidden, dropout=0, channel_wise=False):
        super().__init__()
        self.channel_wise = channel_wise
        if channel_wise:
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.MHA = mha
        self.ff = FeedForward(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        r = x
        if self.channel_wise:
            x_r = self.conv(r.permute(0,2,1)).transpose(1,2)
            k, v = x_r, x_r
        else:
            k, v = r, r
        x, _ = self.MHA(r, k, v, attn_mask=None)
        x = self.norm1(self.dropout(x) + r)
        r = x
        x = self.norm2(self.dropout(self.ff(r)) + r)
        return x


class MultiPatchFormerWrapper(BaseModel):
    """MultiPatchFormer wrapper — seq_len-adaptive patch scales."""

    pipeline = "large"
    uses_internal_normalization = True

    def __init__(self, input_dim, output_dim,
                 d_model=256, n_heads=8, e_layers=3, d_ff=32,
                 dropout=0.1, seq_len=96, pred_len=1, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.pred_len = pred_len
        self.d_channel = input_dim
        self.N = e_layers

        # Patch scale candidates (from paper)
        all_stride = [8, 8, 7, 6]
        all_patch_len = [8, 16, 24, 32]

        # Only keep scales where Conv1d output length > 0
        self.valid_scales = []
        valid_stride = []
        valid_patch_len = []
        for s, pl in zip(all_stride, all_patch_len):
            if (seq_len + s - pl) // s + 1 > 0:
                self.valid_scales.append(len(valid_stride))
                valid_stride.append(s)
                valid_patch_len.append(pl)

        if not self.valid_scales:
            raise ValueError(
                f"seq_len={seq_len} too short for any MultiPatchFormer patch scale. "
                f"Need at least {min(all_patch_len)}.")

        self.stride = valid_stride
        self.patch_len = valid_patch_len
        self.num_scales = len(valid_stride)

        # Compute output patch count per scale (before padding):
        # Scale 0 no padding: (seq_len - pl) // s + 1
        # Other scales: (seq_len + s - pl) // s + 1
        def _pn(i, s, pl):
            return (seq_len + (s if i > 0 else 0) - pl) // s + 1
        pns = [_pn(i, s, pl) for i, (s, pl) in enumerate(zip(self.stride, self.patch_len))]
        self.patch_num = min(pns)  # use min so all scales can produce at least this many

        # Padding layers (only for scales after first)
        self.padding_layers = nn.ModuleList([
            nn.ReplicationPad1d((0, s)) for s in self.stride
        ])

        # Patch embedding output dim (d_model//4 per scale, num_scales scales)
        self.patch_dim = d_model // 4 * self.num_scales
        if self.patch_dim != d_model:
            self.patch_proj = nn.Linear(self.patch_dim, d_model)

        # Positional encoding — patch_num x d_model
        pe = torch.zeros(self.patch_num, d_model)
        for pos in range(self.patch_num):
            for i in range(0, d_model, 2):
                w = 10000 ** ((2 * i) / d_model)
                pe[pos, i] = math.sin(pos / w)
                pe[pos, i + 1] = math.cos(pos / w)
        self.register_buffer("pe", pe.unsqueeze(0))

        # Multi-head attention layers (temporal + channel-wise)
        shared_mha = nn.ModuleList([
            AttentionLayer(FullAttention(), d_model, n_heads) for _ in range(e_layers)
        ])
        self.encoder_list = nn.ModuleList([
            Encoder(d_model, shared_mha[ll], d_ff, dropout, False) for ll in range(e_layers)
        ])
        shared_mha_ch = nn.ModuleList([
            AttentionLayer(FullAttention(), d_model, n_heads) for _ in range(e_layers)
        ])
        self.encoder_list_ch = nn.ModuleList([
            Encoder(d_model, shared_mha_ch[0], d_ff, dropout, True) for _ in range(e_layers)
        ])

        # Channel embedding: input dim depends on patch_dim (after projection to d_model)
        enc_dim = d_model  # after patch_proj, we always have d_model per patch
        self.embedding_channel = nn.Conv1d(enc_dim * self.patch_num, d_model, 1)

        # Multi-scale patch embeddings (one Conv1d per valid scale)
        self.patch_embeds = nn.ModuleList([
            nn.Conv1d(1, d_model // 4, pl, stride=s)
            for pl, s in zip(self.patch_len, self.stride)
        ])

        # Semi auto-regressive output layers
        self.out_layers = nn.ModuleList()
        for i in range(8):
            in_dim = d_model + i * (pred_len // 8)
            out_dim = pred_len // 8 if i < 7 else pred_len - 7 * (pred_len // 8)
            self.out_layers.append(nn.Linear(in_dim, out_dim))

        self.output_proj = nn.Linear(input_dim, output_dim)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        x_i = x_enc.permute(0, 2, 1)  # (B, C, L)
        B, C, L = x_i.shape

        # Multi-scale patch embedding (truncated to uniform patch_num)
        enc_patches = []
        for i in range(self.num_scales):
            x_p = self.padding_layers[i](x_i) if i > 0 else x_i
            x_p = rearrange(x_p, "b c l -> (b c) l").unsqueeze(-1).permute(0, 2, 1)
            x_p = self.patch_embeds[i](x_p).permute(0, 2, 1)  # ((B*C), P, d_model//4)
            enc_patches.append(x_p[:, :self.patch_num, :])  # truncate to min patch_num
        enc = torch.cat(enc_patches, dim=-1)  # ((B*C), patch_num, patch_dim)
        if self.patch_dim != 256:
            enc = self.patch_proj(enc)         # → ((B*C), patch_num, d_model) if needed
        enc = enc + self.pe

        # Temporal encoding (per-patch attention)
        for i in range(self.N):
            enc = self.encoder_list[i](enc)

        # Channel-wise encoding
        x_ch = rearrange(enc, "(b c) p d -> b c (p d)", b=B, c=C)
        x_ch = self.embedding_channel(x_ch.permute(0, 2, 1)).transpose(1, 2)
        for i in range(self.N):
            x_ch = self.encoder_list_ch[i](x_ch)

        # Semi auto-regressive forecast
        forecasts = []
        for i in range(8):
            inp = x_ch if i == 0 else torch.cat([x_ch] + forecasts[:i], dim=-1)
            forecasts.append(self.out_layers[i](inp))

        final = torch.cat(forecasts, dim=-1).permute(0, 2, 1)
        denorm = final * stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len, 1)
        return denorm + means[:, 0].unsqueeze(1).repeat(1, self.pred_len, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        out = out[:, -self.pred_len:, :]
        out = self.output_proj(out)
        return out
