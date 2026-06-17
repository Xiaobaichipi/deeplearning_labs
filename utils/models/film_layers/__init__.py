"""FiLM dependency layers — self-contained package."""
from .HiPPO_LegT import HiPPO_LegT
from .SpectralConv1d import SpectralConv1d

__all__ = [
    "HiPPO_LegT",
    "SpectralConv1d",
]
