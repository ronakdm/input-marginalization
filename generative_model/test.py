import sys
import torch
import torch.nn as nn

sys.path.append("gm-hw1")

import transformer, dataset

device = torch.device("cpu")  # LOCAL

layers = 1
heads = 2
d = 5
k = 3
m = 3

lr = 0.00035
context = 150
batch_size = 32
log_interval = 50
criterion = nn.NLLLoss()

root = "data/wikitext-2"
train_data = dataset.WikiText2(root, context, dataset.DatasetSplit.train)
valid_data = dataset.WikiText2(root, context, dataset.DatasetSplit.valid)

model = transformer.Transformer(
    context, train_data.word_count(), 400, 40, 900, heads, layers, tied_weights=True
).to(device)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True
)


def evaluate(data):
    model.eval()
    with torch.no_grad():
        loss = 0.0
        loader = torch.utils.data.DataLoader(
            dataset=data, batch_size=batch_size, shuffle=False
        )
        for i, (x, y) in enumerate(loader):
            x, y = x.permute(1, 0).to(device), y.permute(1, 0).to(device)
            yhat = model(x).view(-1, train_data.word_count())
            loss += criterion(yhat, y.contiguous().view(-1))

    print()
    model.train()
    return loss / len(loader)


for i, (x, y) in enumerate(train_loader):
    x, y = x.permute(1, 0).to(device), y.permute(1, 0).to(device)
    y_ = model(x)
    val_loss = evaluate(valid_data)
    if i == 3:
        break
