import torch

from model import GPTLanguageModel
from dataset import get_batch, vocab_size, decode

from config import MAX_ITERS, EVAL_INTERVAL, LEARNING_RATE

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = GPTLanguageModel(vocab_size).to(device)

print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            X = X.to(device)
            Y = Y.to(device)
            _, loss = model(X, Y)

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


for iter in range(MAX_ITERS):
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    xb = xb.to(device)
    yb = yb.to(device)
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))
