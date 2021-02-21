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

    def forward(
        self, x, token_type_ids, attention_mask, labels, return_dict=True, train=True
    ):

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

        self.loss = criterion(self.logits, y)

        return self


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_labels, n_rnn_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        ##can change
        
        self.rnn = nn.LSTM(
                           embedding_dim, hidden_dim, num_layers=n_rnn_layers, batch_first=True, bidirectional=True
                           )
                           layered_hidden_dim = hidden_dim * n_rnn_layers * 2
                           self.dropout_train, self.dropout_test, self.dropout_embedded = nn.Dropout(p=0.5), nn.Dropout(p=0), nn.Dropout(p=0.3)
                           self.output = nn.Linear(layered_hidden_dim, n_labels)

    def forward(self, text, train =True):
        embedded = self.embedding(text)
        dropped_embedded = self.dropout_embedded(embedded)
        output, (hidden, cell) = self.rnn(dropped_embedded)

        dropped = (
                   self.dropout_train(hidden) if train else self.dropout_test(hidden)
                   )
            
        dropped = dropped.transpose(0, 1).reshape(hidden.shape[1], -1)
                   
        return self.output(dropped)
