import torch
import math
import torch.nn.functional as F

# from torch.nn import LogSoftmax

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab = tokenizer.vocab
# log_softmax = LogSoftmax(dim=0)


def encode(sentence, device):
    encoded = tokenizer(sentence, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_masks = encoded.attention_mask.to(device)
    labels = torch.tensor([0]).to(device)  # Doesn't matter for computing logits.

    return input_ids, attention_masks, labels


# def compute_log_odds(model, input_ids, attention_masks, labels):
#     logits = model(
#         input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels,
#     ).logits
#     label = torch.argmax(logits[0]).item()
#     log_odds = log_softmax(logits[0])[label] - log_softmax(logits[0])[1 - label]
#     return log_odds


# TODO: Make this work for batches.
def erasure(model, sentence, special_token):
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    input_ids, attention_masks, labels = encode(sentence, device)
    seq_len = input_ids.shape[1]
    model.eval()

    att_scores = torch.zeros(input_ids.shape)
    with torch.no_grad():

        logits_true = model(
            input_ids, attention_mask=attention_masks, labels=labels,
        ).logits[0]

        # TODO: take target label as an argument.
        label = torch.argmax(logits_true)

        for t in range(seq_len):
            temp = input_ids[0, t].item()  # item() to pass by value.
            input_ids[0, t] = vocab[special_token]
            logits = model(
                input_ids, attention_mask=attention_masks, labels=labels,
            ).logits[0]

            att_scores[0, t] = logits_true[label] - logits[label]
            input_ids[0, t] = temp  # Change token back after replacement.

        return att_scores


def zero_erasure(model, sentence):
    return erasure(model, sentence, "[PAD]")


def unk_erasure(model, sentence):
    return erasure(model, sentence, "[UNK]")


def input_marginalization(model, sentence, mlm, num_batches=50):
    input_ids, attention_masks, labels = encode(model, sentence)
    seq_len = input_ids.shape[1]
    model.eval()

    att_scores = torch.zeros(input_ids.shape)
    with torch.no_grad():

        logits_true = model(
            input_ids, attention_mask=attention_masks, labels=labels,
        ).logits[0]

        # TODO: take target label as an argument.
        target_label = torch.argmax(logits_true)

        # Get MLM distribution for every masked word.
        # Shape: [vocab_size * seq_len]
        mlm_logits = mlm(input_ids).logits[0].transpose(0, 1)
        vocab_size = mlm_logits.shape[0]

        expanded_inputs = input_ids.repeat(vocab_size, 1)
        expanded_attns = attention_masks.repeat(vocab_size, 1)
        expanded_labels = labels.repeat(vocab_size)

        vocab_batch_size = math.ceil(vocab_size / num_batches)

        for t in range(seq_len):

            # Get log_prob for every masked word ([vocab_size]).
            # Have to split it into batches by vocab.

            model_log_probs = torch.zeros(vocab_size)

            # Substitute in every word in the vocab for this token.
            temp = torch.tensor(
                [expanded_inputs[word, t].item() for word in range(vocab_size)]
            )
            expanded_inputs[:, t] = torch.arange(vocab_size)

            for b in range(num_batches):
                start_idx = b * vocab_batch_size
                end_idx = min((b + 1) * vocab_batch_size, vocab_size)

                batch_inputs = expanded_inputs[start_idx:end_idx]
                batch_attns = expanded_attns[start_idx:end_idx]
                batch_labels = expanded_labels[start_idx:end_idx]

                # Shape: [vocab_batch_size * num_labels]
                model_logits = model(
                    batch_inputs, attention_mask=batch_attns, labels=batch_labels,
                ).logits

                # Shape: [vocab_batch_size]
                model_log_probs[start_idx:end_idx] = F.log_softmax(model_logits, dim=1)[
                    :, target_label
                ]

            # Shape: [vocab_batch_size]
            mlm_log_probs = F.log_softmax(mlm_logits[:, t], dim=0)
            log_prob_marg = torch.logsumexp(model_log_probs + mlm_log_probs, 0)
            log_odds_marg = log_prob_marg - torch.log(1 - torch.exp(log_prob_marg))

            att_scores[0, t] = logits_true[target_label] - log_odds_marg

            # Replace the tokens that we substituted.
            expanded_inputs[:, t] = temp

        return att_scores


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
