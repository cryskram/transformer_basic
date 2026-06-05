import torch.nn as nn
from attention import (
    MultiHeadAttention,
    FeedForward,
)


class Block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_heads,
    ):
        super().__init__()
        head_size = n_embd // n_heads

        self.sa = MultiHeadAttention(
            n_heads,
            head_size,
        )

        self.ffwd = FeedForward(
            n_embd,
        )

        self.ln1 = nn.LayerNorm(
            n_embd,
        )

        self.ln2 = nn.LayerNorm(
            n_embd,
        )

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
