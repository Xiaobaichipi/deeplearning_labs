"""
Shared Transformer base components.

Current consumers: Informer, VanillaTransformer (and future large-pipeline
models).

NOTE: Autoformer still uses its own independent copies of Encoder/Decoder/
EncoderLayer/DecoderLayer in ``autoformer_layers/Autoformer_EncDec.py``.
These have NOT been migrated to this shared module yet.  A future refactor
(e.g. feat/unify-shared-layers) should make Autoformer import from here
and remove the duplicates.
"""

from .Embed import TokenEmbedding, PositionalEmbedding, DataEmbedding
from .Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer, ConvLayer
from .masking import TriangularCausalMask, ProbMask
from .SelfAttention_Family import FullAttention, AttentionLayer
