import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, vocab_size, embedding_dim):
        super(CNN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(1, 100, (n, embedding_dim)) for n in (3, 4, 5)]
        )
        self.dropout_train, self.dropout_test = nn.Dropout(p=0.5), nn.Dropout(p=0)
        self.linear = nn.Linear(
            in_features=in_channels, out_features=out_channels, bias=True
        )

    def forward(self, x, train=True):

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

        logits = nn.functional.log_softmax(output, dim=1)
        return logits

    # def predict(self, x):
    #     logits = self.forward(x, train=False)
    #     return logits.max(1)[1]

    # def train(self, train_dataloader,validation_dataloader, num_epochs=15, learning_rate):
    #     criterion = nn.NLLLoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    #     loss_vec = []

    #     for epoch in (range(num_epochs)):
    #         t0 = time.time()
    #         epoch_loss = 0
    #         for batch in train_dataloader:
    #             x = batch[0].to(device)
    #             y = batch[2].to(device)
    #             optimizer.zero_grad()
    #             y_p = self.forward(x)

    #             loss = criterion(y_p, y)

    #             loss.backward()

    #             optimizer.step()
    #             epoch_loss += loss.data

    #         self.model = model
    #         loss_vec.append(epoch_loss / len(train_dataloader))

    #         training_time = format_time(time.time() - t0)

    #         acc,validation_time = self.validate(validation_dataloader)
    #         print('Epoch {} loss: {} | acc: {}'.format(epoch+1, loss_vec[epoch-1], acc))
    #         print("  Training epcoh {} took: {:}".format(epoch+1, training_time))
    #         self.model = model

    #         training_stats.append(
    #             {
    #             'epoch': epoch + 1,
    #             'Training Loss': loss_vec[epoch-1],
    #             'Valid. Accur.': acc,
    #             'Training Time': training_time,
    #             'Validation Time': validation_time
    #             }
    #         )

    #     plt.plot(range(len(loss_vec)), loss_vec)
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.show()
    #     print('\nModel trained.\n')
    #     self.loss_vec = loss_vec
    #     self.model = model
    #     pickle.dump(training_stats, open(f"{save_dir}/training_stats_{save_filename}.p", "wb"))

    # def test(self, test_dataloader):
    #     t0 = time.time()

    #     upload, trues = [], []
    #     # Update: for kaggle the bucket iterator needs to have batch_size 10
    #     for batch in test_dataloader:

    #         x, y = batch[0].to(device), batch[2].to(device)
    #         probs = self.predict(x)[:len(y)]
    #         upload += list(probs.data)
    #         trues += list(y.data)

    #     correct = sum([1 if i == j else 0 for i, j in zip(upload, trues)])
    #     accuracy = correct / len(trues)
    #     print('Testset Accuracy:', accuracy)
    #     test_time = format_time(time.time()-t0)
    #     test_stats = {

    #     'Test Accur.': accuracy,
    #     'Test Time': test_time,
    #     }
    #     pickle.dump(test_stats, open(f"{save_dir}/test_stats_{save_filename}.p", "wb"))

    # def validate(self, val_iter):
    #     y_p, y_t, correct = [], [], 0
    #     t0 = time.time()
    #     for batch in val_iter:
    #         x, y = batch[0].to(device), batch[2].to(device)
    #         probs = self.predict(x)[:len(y)]

    #         y_p += list(probs.data)
    #         y_t += list(y.data)

    #     correct = sum([1 if i == j else 0 for i, j in zip(y_p, y_t)])
    #     accuracy = correct / len(y_p)
    #     validation_time = format_time(time.time() - t0)
    #     return accuracy,validation_time
