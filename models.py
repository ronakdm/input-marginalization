import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, vocab_size, embedding_dim):
        super(CNN, self).__init__()
        self.loss = 0
        self.logits = 0
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(1, 100, (n, embedding_dim)) for n in (3, 4, 5)]
        )
        self.dropout_train, self.dropout_test = nn.Dropout(p=0.5), nn.Dropout(p=0)
        self.linear = nn.Linear(
            in_features=in_channels, out_features=out_channels, bias=True
        )

    def forward(self, x, token_type_ids,attention_mask,labels,return_dict=True):

        embedded = self.embeddings(x)
        embedded = embedded.unsqueeze(1)

        convolved = [
            nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs
        ]

        pooled = [
            nn.functional.max_pool1d(convd, convd.size(2)).squeeze(2)
            for convd in convolved
        ]

        concatted = torch.cat(pooled, 1)

        dropped = (
            self.dropout_train(concatted) if train else self.dropout_test(concatted)
        )

        output = self.linear(dropped)

        self.logits = nn.functional.log_softmax(output, dim=1)

        criterion = nn.NLLLoss()

        y = labels
        
        self.loss = criterion(self.logits,y)       
        
        return self
