"""MICN encoder blocks — MIC layer and SeasonalPrediction."""

import torch
import torch.nn as nn


class MIC(nn.Module):
    """MIC layer to extract local and global features."""
    def __init__(self, feature_size=512, dropout=0.05, decomp_kernel=None,
                 conv_kernel=None, isometric_kernel=None):
        super().__init__()
        if decomp_kernel is None: decomp_kernel = [32]
        if conv_kernel is None: conv_kernel = [24]
        if isometric_kernel is None: isometric_kernel = [18, 6]
        self.conv_kernel = conv_kernel

        self.isometric_conv = nn.ModuleList([
            nn.Conv1d(feature_size, feature_size, kernel_size=i, padding=0, stride=1)
            for i in isometric_kernel
        ])

        self.conv = nn.ModuleList([
            nn.Conv1d(feature_size, feature_size, kernel_size=i,
                      padding=i // 2, stride=i)
            for i in conv_kernel
        ])

        self.conv_trans = nn.ModuleList([
            nn.ConvTranspose1d(feature_size, feature_size, kernel_size=i,
                               padding=0, stride=i)
            for i in conv_kernel
        ])

        from .Autoformer_EncDec import series_decomp
        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = nn.Conv2d(feature_size, feature_size,
                               kernel_size=(len(conv_kernel), 1))

        self.conv1 = nn.Conv1d(feature_size, feature_size * 4, kernel_size=1)
        self.conv2 = nn.Conv1d(feature_size * 4, feature_size, kernel_size=1)
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)
        self.norm = nn.LayerNorm(feature_size)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)

    def conv_trans_conv(self, src, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = src.shape
        x = src.permute(0, 2, 1)

        x1 = self.drop(self.act(conv1d(x)))
        x_down = x1

        zeros = torch.zeros((x_down.shape[0], x_down.shape[1], x_down.shape[2] - 1),
                            device=src.device)
        x_up = torch.cat((zeros, x_down), dim=-1)
        x_up = self.drop(self.act(isometric(x_up)))
        x_up = self.norm((x_up + x1).permute(0, 2, 1)).permute(0, 2, 1)

        x_up = self.drop(self.act(conv1d_trans(x_up)))
        x_up = x_up[:, :, :seq_len]
        x_up = self.norm(x_up.permute(0, 2, 1) + src)
        return x_up

    def forward(self, src):
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, trend1 = self.decomp[i](src)
            src_out = self.conv_trans_conv(
                src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i])
            multi.append(src_out)

        mg = torch.cat([multi[i].unsqueeze(1) for i in range(len(self.conv_kernel))], dim=1)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)

        y = self.norm1(mg)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)
        return self.norm2(mg + y)


class SeasonalPrediction(nn.Module):
    """Seasonal component prediction with stacked MIC layers."""
    def __init__(self, embedding_size=512, dropout=0.05, d_layers=1,
                 decomp_kernel=None, conv_kernel=None, isometric_kernel=None,
                 c_out=1):
        super().__init__()
        if decomp_kernel is None: decomp_kernel = [32]
        if conv_kernel is None: conv_kernel = [24]
        if isometric_kernel is None: isometric_kernel = [18, 6]

        self.mic = nn.ModuleList([
            MIC(feature_size=embedding_size, dropout=dropout,
                decomp_kernel=decomp_kernel, conv_kernel=conv_kernel,
                isometric_kernel=isometric_kernel)
            for _ in range(d_layers)
        ])
        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)
