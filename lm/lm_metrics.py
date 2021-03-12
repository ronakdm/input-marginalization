import torch


def loaddata():
    train_dataloader, validation_dataloader, test_dataloader = generate_dataloaders(1)
    return test_dataloader


def compute_probability2(model, input_ids, attention_masks, label):

    logits = model(
        input_ids.to(torch.int64),
        token_type_ids=None,
        attention_mask=attention_masks,
        labels=label.repeat((len(input_ids))),
    ).logits

    return torch.exp(torch.reshape(logits[:, label], (-1,)))


def compute_probability(model, input_ids, attention_masks, label):
    logits = model(
        input_ids,
        token_type_ids=None,
        attention_mask=attention_masks,
        labels=label.repeat((len(input_ids))),
    ).logits

    return math.exp(logits[0][label])


def calculate_woe(model, input_ids, attention_masks, label, sigma):
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    bert_model.to(device)

    # predictions is the probability distribution of each word in the vocabulary for each word in input sentence
    predictions = bert_model(input_ids)
    predictions = torch.squeeze(predictions)
    predictions = F.softmax(predictions, dim=1)

    # woe is the weight of evidence
    woe = []
    model.eval()

    with torch.no_grad():
        for j in range(len(predictions)):
            word_scores = predictions[j]
            input_batch = input_ids.clone().to(device)

            # word_scores_batch calculates the value of the MLM of Bert for each masked word
            # we put 0 for the first input which is unmasked
            word_scores_batch = [0]

            for k in range(len(word_scores)):
                if word_scores[k] > sigma:
                    input_batch = torch.cat((input_batch, input_ids), 0)
                    input_batch[len(input_batch) - 1][j] = k
                    word_scores_batch.append(word_scores[k].item())

            # probability_input calculates the p(label|sentence) of the target model given each masked input sentence
            probability_input = compute_probability2(
                model, input_batch, attention_masks, label
            )

            m = torch.dot(torch.tensor(word_scores_batch).to(device), probability_input)
            logodds_input = math.log(probability_input[0] / (1 - probability_input[0]))
            logodds_m = math.log(m / (1 - m))
            woe.append(logodds_input - logodds_m)
    return woe

