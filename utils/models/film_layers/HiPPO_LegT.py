"""HiPPO-LegT — Legendre polynomial state-space projection.

Adapted from ``time_series_models_labs/models/FiLM.py``.
Removed hardcoded CUDA device — tensor device is inferred from input.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy import special as ss


def transition(N):
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1)[:, None]
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1, (-1.) ** (i - j + 1)) * R
    B = (-1.) ** Q[:, None] * R
    return A, B


class HiPPO_LegT(nn.Module):
    """HiPPO Legendre projection — projects input onto Legendre polynomial basis.

    Parameters
    ----------
    N : int
        Order of the HiPPO projection (window_size).
    dt : float
        Discretization step size — should be roughly inverse to sequence length.
    discretization : str
        Discretization method for cont2discrete (default: 'bilinear').

    Input:  (length, batch, features)
    Output: (length, batch, features, N)
    """

    def __init__(self, N, dt=1.0, discretization='bilinear'):
        super().__init__()
        self.N = N
        A, B = transition(N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)
        B = B.squeeze(-1)

        # Register as buffers — device will match input at runtime
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))
        self.register_buffer('B', torch.tensor(B, dtype=torch.float32))
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer(
            'eval_matrix',
            torch.tensor(ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T,
                         dtype=torch.float32),
        )

    def forward(self, inputs):
        """Project inputs onto Legendre polynomial basis.

        Parameters
        ----------
        inputs : (length, batch, features)

        Returns
        -------
        (length, batch, features, N)
        """
        c = torch.zeros(inputs.shape[:-1] + (self.N,), device=inputs.device)
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            new = f @ self.B.unsqueeze(0)
            c = F.linear(c, self.A) + new
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        """Reconstruct from Legendre coefficients."""
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)
