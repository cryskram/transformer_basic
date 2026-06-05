import torch
from config import BLOCK_SIZE, BATCH_SIZE

with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

split = int(0.9 * len(data))

train_data = data[:split]
val_data = data[split:]


def get_batch(split):
    source = train_data if split == "train" else val_data

    ix = torch.randint(len(source) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([source[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1 : i + BLOCK_SIZE + 1] for i in ix])

    return x, y


if __name__ == "__main__":

    print("Vocabulary Size:", vocab_size)

    x, y = get_batch("train")

    print(x.shape)
    print(y.shape)
