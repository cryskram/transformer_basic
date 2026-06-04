import torch

with open("data/shakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encoder = lambda s: [stoi[c] for c in s]
decoder = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encoder(text), dtype=torch.long)

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

print(size)
print(data[:20])

BLOCK_SIZE = 128
BATCH_SIZE = 32


def get_batch(split):
    source = train_data if split == "train" else val_data

    ix = torch.randint(len(source) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([source[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1 : i + BLOCK_SIZE + 1] for i in ix])

    return x, y


x, y = get_batch("train")

print(x.shape)
print(y.shape)
