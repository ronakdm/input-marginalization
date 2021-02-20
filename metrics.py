import torch
from torch.nn import LogSoftmax

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab = tokenizer.vocab
log_softmax = LogSoftmax(dim=0)


def encode(model, sentence):
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    encoded = tokenizer(sentence, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_masks = encoded.attention_mask.to(device)
    labels = torch.tensor([0]).to(device)  # Doesn't matter for computing logits.

    return input_ids, attention_masks, labels


def compute_log_odds(model, input_ids, attention_masks, labels):
    logits = model(
        input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels,
    ).logits
    label = torch.argmax(logits[0]).item()
    log_odds = log_softmax(logits[0])[label] - log_softmax(logits[0])[1 - label]
    return log_odds


# TODO: Make this work for batches.
def erasure(model, sentence, special_token):
    input_ids, attention_masks, labels = encode(model, sentence)
    seq_len = input_ids.shape[1]
    model.eval()

    att_scores = torch.zeros(input_ids.shape)
    with torch.no_grad():

        log_odds_true = compute_log_odds(model, input_ids, attention_masks, labels)

        for t in range(seq_len):
            token = input_ids[0, t].item()  # item() to pass by value.
            input_ids[0, t] = vocab[special_token]
            log_odds = compute_log_odds(model, input_ids, attention_masks, labels)

            att_scores[0, t] = log_odds_true - log_odds
            input_ids[0, t] = token  # Change token back after replacement.

        return att_scores


def zero_erasure(model, sentence):
    return erasure(model, sentence, "[PAD]")


def unk_erasure(model, sentence):
    return erasure(model, sentence, "[UNK]")


# TODO: Input marginalization
