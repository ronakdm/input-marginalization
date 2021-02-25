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

        # log_odds_true = compute_log_odds(model, input_ids, attention_masks, labels)
        logits_true = model(
            input_ids,
            token_type_ids=None,
            attention_mask=attention_masks,
            labels=labels,
        ).logits[0]

        # TODO: Check which one is correct.
        label = torch.argmax(logits_true)

        for t in range(seq_len):
            token = input_ids[0, t].item()  # item() to pass by value.
            input_ids[0, t] = vocab[special_token]
            # log_odds = compute_log_odds(model, input_ids, attention_masks, labels)
            logits = model(
                input_ids,
                token_type_ids=None,
                attention_mask=attention_masks,
                labels=labels,
            ).logits[0]

            att_scores[0, t] = logits_true[label] - logits[label]
            input_ids[0, t] = token  # Change token back after replacement.

        return att_scores


def zero_erasure(model, sentence):
    return erasure(model, sentence, "[PAD]")


def unk_erasure(model, sentence):
    return erasure(model, sentence, "[UNK]")


def color_sentence(model, sentence, erasure_type):
    evaluate_tensor = erasure_type(model, sentence)[0][
        1:-1
    ]  # extra characters been removed

    # define some color for different levels of effect
    dark_red = [150, 0, 0]
    red = [225, 0, 0]
    orange = [255, 160, 100]
    dark_blue = [0, 50, 180]
    blue = [0, 150, 225]
    light_blue = [180, 240, 255]

    splits = [-0.2, -0.1, -0.05, 0.3, 0.5, 1]

    colored = []

    for i in range(len(evaluate_tensor)):

        if evaluate_tensor[i].item() > splits[5]:  # very positive
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(dark_red[0]),
                    str(dark_red[1]),
                    str(dark_red[2]),
                    sentence.split()[i],
                )
            )
        elif evaluate_tensor[i].item() > splits[4]:
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(red[0]), str(red[1]), str(red[2]), sentence.split()[i]
                )
            )
        elif evaluate_tensor[i].item() > splits[3]:
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(orange[0]), str(orange[1]), str(orange[2]), sentence.split()[i]
                )
            )
        elif evaluate_tensor[i].item() < splits[0]:  # very negative
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(dark_blue[0]),
                    str(dark_blue[1]),
                    str(dark_blue[2]),
                    sentence.split()[i],
                )
            )
        elif evaluate_tensor[i].item() < splits[1]:
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(blue[0]), str(blue[1]), str(blue[2]), sentence.split()[i]
                )
            )
        elif evaluate_tensor[i].item() < splits[2]:
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(light_blue[0]),
                    str(light_blue[1]),
                    str(light_blue[2]),
                    sentence.split()[i],
                )
            )
        else:
            colored.append(sentence.split()[i])

    print(" ".join([str(elem) for elem in colored]))


# TODO: Input marginalization
