import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import GPT2Tokenizer


class GPT2TextDataset(Dataset):
    def __init__(self, file_path="data/wikitext-2-raw/wiki.train.raw", seq_len=512):

        # Use GPT-2 vocabulary.
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Read text file as one long string, and tokenize into a list of vocab indices.
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        tokenized_text = tokenizer(text).input_ids

        # Chop up the tokenized text into seq_len-sized windows.
        self.examples = []
        for i in range(0, len(tokenized_text) - seq_len + 1, seq_len):
            self.examples.append(tokenized_text[i : i + seq_len])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def make_dataloaders(batch_size):
    dataloaders = []
    for dataset_name in ["train", "valid", "test"]:
        dataset_path = (
            "input-marginalization/data/wikitext-2-raw/wiki.%s.raw" % dataset_name
        )
        dataset = GPT2TextDataset(file_path=dataset_path)
        dataloader = DataLoader(
            dataset, sampler=SequentialSampler(dataset), batch_size=batch_size
        )
        print("{:>5,} {} samples.".format(len(dataset), dataset_name))

        dataloaders.append(dataloader)

    return dataloaders
