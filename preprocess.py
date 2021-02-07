import pickle
import torch
from transformers import BertTokenizer

id_sentence = pickle.load(open("data/id_sentence.p", "rb"))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def tokensize_and_save_SST2(dataset):
    filename = "data/id_rating_%s.p" % dataset
    id_rating = pickle.load(open(filename, "rb"))

    labels = []
    sentences = []

    # Collect only sentences that do not have `neural` sentiment.
    for key in id_rating:
        if id_rating[key] >= 0.6:
            labels.append(1)
            sentences.append(" ".join(id_sentence[key]))
        elif id_rating[key] <= 0.4:
            labels.append(0)
            sentences.append(" ".join(id_sentence[key]))

    encoded = tokenizer(sentences, padding="longest", return_tensors="pt",)
    input_ids = encoded.input_ids
    attention_masks = encoded.attention_mask
    labels = torch.tensor(labels)

    print("Number of %s sentences:" % dataset, input_ids.shape[0])
    print("Maximum %s sequence length:" % dataset, input_ids.shape[1])

    pickle.dump(
        input_ids, open("preprocessed_data/SST-2/input_ids_%s" % dataset, "wb"),
    )
    pickle.dump(
        attention_masks,
        open("preprocessed_data/SST-2/attention_masks_%s" % dataset, "wb"),
    )
    pickle.dump(labels, open("preprocessed_data/SST-2/labels_%s" % dataset, "wb"))


for dataset in ["train", "valid", "test"]:
    tokensize_and_save_SST2(dataset)

