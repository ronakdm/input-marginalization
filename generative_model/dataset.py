import os,enum

import torch

DatasetSplit = enum.Enum('DatasetSplit', 'train valid test')

class WikiText2(torch.utils.data.Dataset):
    """
        PyTorch Dataset for the WikiText2 corpus:
            https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
    """

    def __init__(self, root, context, split, block=True):
        self.context = context
        self.block = block
        self.word2idx = {}
        self.idx2word = []

        # build the vocabulary from the training data
        self._tokenize(os.path.join(root, 'train.txt'))
        self.data = self._tokenize(os.path.join(root, split.name + '.txt'))

    def __len__(self):
        if self.block:
            return len(self.data) // self.context
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.block:
            x = self.data[idx*self.context:(idx+1)*self.context]
            y = self.data[idx*self.context+1:(idx+1)*self.context+1].view(-1)
        else:    
            x = torch.tensor([self.word2idx['<pad>']] * self.context)
            y = torch.tensor([self.word2idx['<pad>']] * self.context)
            context = min(self.context,idx)
            if idx > 0: x[-context:] = self.data[idx-context:idx]
            context = min(self.context,idx+1)
            y[-context:] = self.data[idx+1-context:idx+1]

        return x, y
        
    def word_count(self):
        # don't count <pad> as a word
        return len(self.idx2word)

    def _add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def _tokenize(self, path):
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    self._add_word(word)
                    ids.append(self.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))

        self._add_word('<pad>')
        return torch.cat(idss)

