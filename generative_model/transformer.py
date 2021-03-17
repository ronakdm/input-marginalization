import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, heads, d, k, m, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.k = k
        self.heads = heads

        self.wq = nn.Linear(d, heads * k, bias=False)
        self.wk = nn.Linear(d, heads * k, bias=False)
        self.wv = nn.Linear(d, heads * k, bias=False)
        self.wc = nn.Linear(heads * k, d, bias=False)
        self.dropoutatt = nn.Dropout(dropout)

        self.w1 = nn.Linear(d, m)
        self.dropoutfc = nn.Dropout(dropout)
        self.w2 = nn.Linear(m, d)

        self.layernorm1 = nn.LayerNorm(d)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d)
        self.dropout2 = nn.Dropout(dropout)

        nn.init.normal_(self.wq.weight, 0, 0.02)
        nn.init.normal_(self.wk.weight, 0, 0.02)
        nn.init.normal_(self.wv.weight, 0, 0.02)
        nn.init.normal_(self.wc.weight, 0, 0.02)

        nn.init.normal_(self.w1.weight, 0, 0.02)
        nn.init.constant_(self.w1.bias, 0.0)
        nn.init.normal_(self.w2.weight, 0, 0.02)
        nn.init.constant_(self.w2.bias, 0.0)

    def forward(self, x, mask):
        seq_len, batch_size, embed_dim = x.shape

        # Hint: Writing efficient code is almost as important as
        # writing correct code in ML.
        # Avoid writing for-loops! Consider using the
        # batch matrix multiplication operator torch.bmm
        #  raise NotImplementedError('Implement a transformer block')

        p = seq_len
        n = batch_size
        k = self.k
        h = self.heads

        # Move batch to the front. Now [n*p*d].
        x_ = x.permute(1, 0, 2)

        # Compute query, key, and value for each head.
        Q = self.wq(x_).view((n, p, h, k)).permute(0, 2, 1, 3)  # [n*h*p*k]
        K = self.wk(x_).view((n, p, h, k)).permute(0, 2, 3, 1)  # [n*h*k*p]
        V = self.wv(x_).view((n, p, h, k)).permute(0, 2, 1, 3)  # [n*h*p*k]

        # Compute attention (dot over k). # Mask is [p*p]
        dot_prod = torch.matmul(Q, K) / math.sqrt(k)
        A = self.dropoutatt(nn.Softmax(dim=3)(dot_prod + mask))  # [n*h*p*p]

        # Compute attention * value.
        u = torch.matmul(A, V)  # [n*h*p*k]
        u = u.permute(0, 2, 1, 3).reshape((n, p, h * k))  # [n*p*(h*k)]
        u = self.wc(u)  # [n*p*d]

        u = self.layernorm1(self.dropoutfc(u) + x_)
        z = self.w2(self.dropout1(F.relu(self.w1(u))))
        out = self.layernorm2(self.dropout2(z) + u)

        return out.permute(1, 0, 2)


class Transformer(nn.Module):
    def __init__(
        self,
        seq_len,
        tokens,
        d,
        k,
        m,
        heads,
        layers,
        tied_weights=False,
        dropout=0.0,
        dropoutio=0.0,
    ):
        super(Transformer, self).__init__()
        self.mask = None
        self.pos = None
        self.dims = d
        self.tied_weights = tied_weights
        self.dropout = dropout

        self.positional_embedding = nn.Embedding(seq_len, d)
        self.dropi = nn.Dropout(dropoutio)
        self.word_embedding = nn.Embedding(tokens, d)
        self.transformer = nn.ModuleList()
        for i in range(layers):
            self.transformer.append(TransformerBlock(heads, d, k, m, dropout))

        if not tied_weights:
            self.decoder = nn.Linear(d, tokens)
        self.dropo = nn.Dropout(dropoutio)
        self.bias = nn.Parameter(torch.ones(tokens))

        nn.init.normal_(self.positional_embedding.weight, 0, 0.02)
        nn.init.normal_(self.word_embedding.weight, 0, 0.02)
        if not self.tied_weights:
            nn.init.normal_(self.decoder.weight, 0, 0.02)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        if self.mask is None or self.mask.shape[0] != x.shape[0]:
            self.mask = torch.triu(torch.ones(len(x), len(x)))
            self.mask.masked_fill_(self.mask == 0, float("-inf")).masked_fill_(
                self.mask == 1, float(0.0)
            )
            self.mask = self.mask.transpose(0, 1).to(x.device)
            self.pos = torch.arange(0, x.shape[0], dtype=torch.long).to(x.device)

        x = self.word_embedding(x) * math.sqrt(self.dims)
        p = self.positional_embedding(self.pos)[:, None, :]
        z = F.relu(self.dropi(x) + self.dropi(p))
        for layer in self.transformer:
            z = layer(z, self.mask)

        z = self.dropo(z)
        outputs = (
            torch.matmul(z, self.word_embedding.weight.t())
            if self.tied_weights
            else self.decoder(z)
        )
        return F.log_softmax(outputs + self.bias, dim=-1)
