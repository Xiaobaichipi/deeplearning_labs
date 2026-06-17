"""FEDformer dependency layers — shares Encoder/Decoder/AutoCorrelationLayer with Autoformer."""

from ..autoformer_layers.Autoformer_EncDec import (
    Encoder, Decoder, EncoderLayer, DecoderLayer,
    my_Layernorm, series_decomp,
)
from ..autoformer_layers.AutoCorrelation import AutoCorrelationLayer

from .FourierCorrelation import FourierBlock, FourierCrossAttention
from .MultiWaveletCorrelation import MultiWaveletTransform, MultiWaveletCross
