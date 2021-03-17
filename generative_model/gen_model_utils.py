import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import BertTokenizer


class WikiText2Dataset(Dataset):
    def __init__(self, file_path="data/wikitext-2-raw/wiki.train.raw", seq_len=512):

        self.context = seq_len

        # Use BERT vocabulary.
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Read text file as one long string, and tokenize into a list of vocab indices.
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        # tokenized_text = self.tokenizer(text).input_ids
        self.data = torch.tensor(self.tokenizer(text).input_ids, dtype=torch.int64)

        # Chop up the tokenized text into seq_len-sized windows.
        # self.examples = []
        # for i in range(0, len(tokenized_text) - seq_len + 1, seq_len):
        #     self.examples.append(tokenized_text[i : i + seq_len])

    def __len__(self):
        # return len(self.examples)
        return len(self.data) // self.context

    def __getitem__(self, idx):
        x = self.data[idx * self.context : (idx + 1) * self.context]
        y = self.data[idx * self.context + 1 : (idx + 1) * self.context + 1].view(-1)

        # return torch.tensor(self.examples[item])
        return x, y

    def word_count(self):
        # Don't count [PAD]
        return len(self.tokenizer.vocab) - 1


def make_dataloaders(batch_size):
    dataloaders = []
    for dataset_name in ["train", "valid", "test"]:
        dataset_path = (
            "input-marginalization/data/wikitext-2-raw/wiki.%s.raw" % dataset_name
        )
        dataset = WikiText2Dataset(file_path=dataset_path)
        dataloader = DataLoader(
            dataset, sampler=SequentialSampler(dataset), batch_size=batch_size
        )
        print("{:>5,} {} samples.".format(len(dataset), dataset_name))

        dataloaders.append(dataloader)

    return dataloaders
