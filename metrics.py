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
def erasure(model, sentence, special_token, target_label=None):
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    input_ids, attention_masks, labels = encode(sentence, device)
    seq_len = input_ids.shape[1]
    model.eval()

    att_scores = torch.zeros(input_ids.shape)
    with torch.no_grad():

        logits_true = model(
            input_ids, attention_mask=attention_masks, labels=labels,
        ).logits[0]

        if target_label is None:
            target_label = torch.argmax(logits_true)

        for t in range(seq_len):
            temp = input_ids[0, t].item()  # item() to pass by value.
            input_ids[0, t] = vocab[special_token]
            logits = model(
                input_ids, attention_mask=attention_masks, labels=labels,
            ).logits[0]

            att_scores[0, t] = logits_true[target_label] - logits[target_label]
            input_ids[0, t] = temp  # Change token back after replacement.

        return att_scores


def zero_erasure(model, sentence):
    return erasure(model, sentence, "[PAD]")


def unk_erasure(model, sentence):
    return erasure(model, sentence, "[UNK]")


def input_marginalization(model, sentence, mlm, target_label=None, num_batches=50):
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    input_ids, attention_masks, labels = encode(sentence, device)
    seq_len = input_ids.shape[1]
    model.eval()

    att_scores = torch.zeros(input_ids.shape)
    with torch.no_grad():

        logits_true = model(
            input_ids, attention_mask=attention_masks, labels=labels,
        ).logits[0]

        if target_label is None:
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

            model_log_probs = torch.zeros(vocab_size).to(device)

            # Substitute in every word in the vocab for this token.
            temp = input_ids[0, t].item()
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
            expanded_inputs[:, t] = torch.full((vocab_size,), temp)

        return att_scores


def score_to_color(score, color_limit):

    lowest = -color_limit
    highest = color_limit

    if score > highest:
        rgb = [255, 0, 0]
    elif score < lowest:
        rgb = [0, 0, 255]
    elif score > 0:
        frac = (highest - score) / highest
        red = 255
        blue = int(255 * (1 - frac))
        green = int(255 * (1 - frac))

        rgb = [red, blue, green]
    elif score < 0:
        frac = (lowest - score) / lowest
        blue = 255
        red = int(255 * (1 - frac))
        green = int(255 * (1 - frac))

        rgb = [red, blue, green]
    else:
        rgb = [255, 255, 255]

    return str(rgb[0]), str(rgb[1]), str(rgb[2])


def continuous_colored_sentence(sentence, att_scores, color_limit=3):

    input_ids, _, _ = encode(sentence, "cpu")
    tokenized_sentence = tokenizer.convert_ids_to_tokens(input_ids[0, 1:-1])
    scores = att_scores[0]

    colored = []
    joined = []

    for i in range(len(tokenized_sentence)):
        if tokenized_sentence[i][0] == "#":
            tokenized_sentence[i] = tokenized_sentence[i][2:]
            joined.append(1)
        else:
            joined.append(0)

        colors = score_to_color(scores[i], color_limit)
        colored.append(
            "\033[48;2;{};{};{}m{}\033[0m".format(
                colors[0], colors[1], colors[2], tokenized_sentence[i],
            )
        )
    sent = ""

    for i, elem in enumerate(colored):
        if joined[i] == 1:
            sent = sent + str(elem)
        else:
            sent = sent + " " + str(elem)

    print(sent)


def colored_sentence(sentence, att_scores):

    input_ids, _, _ = encode(sentence, "cpu")
    tokenized_sentence = tokenizer.convert_ids_to_tokens(input_ids[0, 1:-1])
    scores = att_scores[0]

    # Define some color for different levels of effect.
    red3 = [255, 0, 0]
    red2 = [225, 102, 102]
    red1 = [255, 204, 204]
    red0 = [255, 230, 234]
    blue0 = [204, 229, 255]
    blue1 = [204, 229, 255]
    blue2 = [102, 178, 225]
    blue3 = [0, 0, 255]

    splits = [-0.2, -0.1, -0.05, 0, 0.3, 0.5, 1]

    colored = []
    joined = []

    for i in range(len(tokenized_sentence)):
        if tokenized_sentence[i][0] == "#":
            tokenized_sentence[i] = tokenized_sentence[i][2:]
            joined.append(1)
        else:
            joined.append(0)

        if scores[i] > splits[6]:  # very positive
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(red3[0]), str(red3[1]), str(red3[2]), tokenized_sentence[i],
                )
            )
        elif scores[i] > splits[5]:
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(red2[0]), str(red2[1]), str(red2[2]), tokenized_sentence[i]
                )
            )
        elif scores[i] > splits[4]:
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(red1[0]), str(red1[1]), str(red1[2]), tokenized_sentence[i]
                )
            )
        elif scores[i] > splits[3]:
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(red0[0]), str(red0[1]), str(red0[2]), tokenized_sentence[i]
                )
            )
        elif scores[i] > splits[2]:
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(blue0[0]), str(blue0[1]), str(blue0[2]), tokenized_sentence[i]
                )
            )
        elif scores[i] > splits[1]:
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(blue1[0]), str(blue1[1]), str(blue1[2]), tokenized_sentence[i]
                )
            )
        elif scores[i] > splits[0]:
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(blue2[0]), str(blue2[1]), str(blue2[2]), tokenized_sentence[i]
                )
            )

        else:
            colored.append(
                "\033[48;2;{};{};{}m{}\033[0m".format(
                    str(blue3[0]), str(blue3[1]), str(blue3[2]), tokenized_sentence[i]
                )
            )
    sent = ""

    for i, elem in enumerate(colored):
        if joined[i] == 1:
            sent = sent + str(elem)
        else:
            sent = sent + " " + str(elem)

    print(sent)

