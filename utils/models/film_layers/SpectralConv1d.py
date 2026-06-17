"""SpectralConv1d — 1D Fourier convolution layer.

Adapted from ``time_series_models_labs/models/FiLM.py``.
FFT → linear transform on selected modes → IFFT.
"""

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """1D Fourier convolution layer.

    Performs FFT, linear transform on selected frequency modes, then IFFT.

    Parameters
    ----------
    in_channels : int
        Input channel dimension.
    out_channels : int
        Output channel dimension.
    seq_len : int
        Sequence length (used to determine max modes).
    ratio : float
        Fraction of frequency modes to keep (default: 0.5).
        ``modes = min(32, seq_len // 2)`` regardless of ratio.
    """

    def __init__(self, in_channels, out_channels, seq_len, ratio=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.modes = min(32, seq_len // 2)
        self.index = list(range(0, self.modes))

        self.scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))
        self.weights_imag = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))

    def compl_mul1d(self, order, x, weights_real, weights_imag):
        return torch.complex(
            torch.einsum(order, x.real, weights_real) - torch.einsum(order, x.imag, weights_imag),
            torch.einsum(order, x.real, weights_imag) + torch.einsum(order, x.imag, weights_real),
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : (B, H, E, N)

        Returns
        -------
        (B, H, out_channels, N)
        """
        B, H, E, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, H, self.out_channels, x.size(-1) // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        a = x_ft[:, :, :, :self.modes]
        out_ft[:, :, :, :self.modes] = self.compl_mul1d(
            "bjix,iox->bjox", a, self.weights_real, self.weights_imag)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
