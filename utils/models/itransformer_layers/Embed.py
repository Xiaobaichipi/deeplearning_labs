import torch
import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    """Inverted embedding — treats variates as tokens, time as features.

    Projects ``c_in`` (number of features) to ``d_model`` via Linear.
    If ``x_mark`` is provided, it is concatenated along the variate dimension
    before the projection, effectively increasing ``c_in``.
    """
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x: [Batch, Time, Variate]  → permute → [Batch, Variate, Time]
        x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # [Batch, Variate, d_model]
        return self.dropout(x)
