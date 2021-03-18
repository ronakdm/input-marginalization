import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
criterion = nn.NLLLoss()


def encode(sentence, device):
    encoded = tokenizer(sentence, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_masks = encoded.attention_mask.to(device)
    labels = torch.tensor([0]).to(device)  # Doesn't matter for computing logits.

    return input_ids, attention_masks, labels


# Take a [batch_size * seq_len] group of sentences and give the log probabilities.
def compute_log_probs(model, input_ids, device):
    model.eval()

    batch_size, seq_len = input_ids.shape

    log_probs = torch.zeros(batch_size)
    with torch.no_grad():
        for i in range(batch_size):
            x = input_ids[i, 0 : seq_len - 1].reshape(1, -1)
            y = input_ids[i, 1:].view(-1).reshape(1, -1)

            x, y = x.permute(1, 0).to(device), y.permute(1, 0).to(device)
            yhat = model(x).view(-1, len(tokenizer.vocab) - 1)
            log_probs[i] = -criterion(yhat, y.contiguous().view(-1))

    return log_probs


def input_marginalization(model, sentence, mlm, num_batches=50):
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"

    input_ids, attention_masks, labels = encode(sentence, device)

    seq_len = input_ids.shape[1]
    model.eval()

    att_scores = torch.zeros(input_ids.shape)

    with torch.no_grad():

        # logits_true = model(
        #     input_ids, attention_mask=attention_masks, labels=labels,
        # ).logits[0]
        log_probs_true = compute_log_probs(model, input_ids, device)[0]
        log_odds_true = log_probs_true - torch.log(1 - torch.exp(log_probs_true))

        # Get MLM distribution for every masked word.
        # Shape: [vocab_size * seq_len]

        mlm_logits = mlm(input_ids).logits[0].transpose(0, 1)

        vocab_size = mlm_logits.shape[0]

        expanded_inputs = input_ids.repeat(vocab_size, 1)
        # expanded_attns = attention_masks.repeat(vocab_size, 1)
        # expanded_labels = labels.repeat(vocab_size)

        vocab_batch_size = vocab_size // num_batches

        for t in range(seq_len):

            # Get log_prob for every masked word ([vocab_size]).
            # Have to split it into batches by vocab.
            print("Word:", t, "-----------------------------")
            model_log_probs = torch.zeros(vocab_size).to(device)

            # Substitute in every word in the vocab for this token.
            temp = input_ids[0, t].item()
            expanded_inputs[:, t] = torch.arange(vocab_size)

            for b in range(num_batches):
                print("Batch:", b)
                start_idx = b * vocab_batch_size
                end_idx = min((b + 1) * vocab_batch_size, vocab_size)

                batch_inputs = expanded_inputs[start_idx:end_idx]
                # batch_attns = expanded_attns[start_idx:end_idx]
                # batch_labels = expanded_labels[start_idx:end_idx]

                # Shape: [vocab_batch_size * num_labels]
                model_log_probs[start_idx:end_idx] = compute_log_probs(
                    model, batch_inputs, device
                )

            # Shape: [vocab_batch_size]
            mlm_log_probs = F.log_softmax(mlm_logits[:, t], dim=0)
            log_prob_marg = torch.logsumexp(model_log_probs + mlm_log_probs, 0)
            log_odds_marg = log_prob_marg - torch.log(1 - torch.exp(log_prob_marg))

            att_scores[0, t] = log_odds_true - log_odds_marg

            # Replace the tokens that we substituted.
            expanded_inputs[:, t] = torch.full((vocab_size,), temp)

        return att_scores
