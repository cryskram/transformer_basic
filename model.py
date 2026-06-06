import torch
import torch.nn as nn

from transformer import Block

from config import (
    BLOCK_SIZE,
    N_EMBD,
    N_HEADS,
    N_LAYERS,
    VOCAB_SIZE,
)


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            N_EMBD,
        )

        self.position_embedding_table = nn.Embedding(
            BLOCK_SIZE,
            N_EMBD,
        )

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=N_EMBD,
                    n_heads=N_HEADS,
                )
                for _ in range(N_LAYERS)
            ]
        )

        self.ln_f = nn.LayerNorm(
            N_EMBD,
        )

        self.lm_head = nn.Linear(
            N_EMBD,
            vocab_size,
        )

    def forward(
        self,
        idx,
        targets=None,
    ):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)

        pos_emb = self.position_embedding_table(
            torch.arange(
                T,
                device=idx.device,
            )
        )

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(
                B * T,
                C,
            )

            targets = targets.view(B * T)

            loss = nn.functional.cross_entropy(
                logits,
                targets,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):

            idx_cond = idx[
                :,
                -BLOCK_SIZE:,
            ]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            logits = logits / temperature

            probs = torch.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat(
                (
                    idx,
                    next_token,
                ),
                dim=1,
            )

        return idx


model = GPTLanguageModel(VOCAB_SIZE)

print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
