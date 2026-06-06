import torch
import torch.nn as nn

from config import (
    N_EMBD,
    BLOCK_SIZE,
)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(
            N_EMBD,
            head_size,
            bias=False,
        )

        self.query = nn.Linear(
            N_EMBD,
            head_size,
            bias=False,
        )

        self.value = nn.Linear(
            N_EMBD,
            head_size,
            bias=False,
        )

        self.dropout = nn.Dropout(0.2)

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                )
            ),
        )

        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf"),
        )

        wei = torch.softmax(
            wei,
            dim=-1,
        )

        wei = self.dropout(wei)

        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_size,
    ):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        self.proj = nn.Linear(
            num_heads * head_size,
            N_EMBD,
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1,
        )

        out = self.proj(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                n_embd,
                4 * n_embd,
            ),
            nn.ReLU(),
            nn.Linear(
                4 * n_embd,
                n_embd,
            ),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)
