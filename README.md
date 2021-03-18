# input-marginalization
**Joint work with [Xinyu Ma](https://github.com/maxinY), [Mayura Patwardhan](https://github.com/mayapatward), and [Peter Michael](https://github.com/ptrmcl)**. This repo contains code and data to implement ["Interpretation of NLP models through input marginalization"](https://www.aclweb.org/anthology/2020.emnlp-main.255/) by Kim *et al*.

## Data

One of the datasets used for experimentation is the [Stanford Sentiment Treebank (SST-2)](https://www.kaggle.com/atulanandjha/stanford-sentiment-treebank-v2-sst2). We started with the SST-2 [sentences](https://github.com/frankaging/SST2-Sentence) cleaned by [frankaging](https://github.com/frankaging), which are in the `data` folder. We then preprocessed the data by removing "neutral" sentiment sentences and representing the sentences in the [BERT](https://huggingface.co/transformers/model_doc/bert.html) vocabulary, which can be found in `preprocessed_data`.

The other dataset used is the [Stanford Natural Language Inference (SNLI) Corpus](https://nlp.stanford.edu/projects/snli/). We processed this data into a tokenized representation using the BERT tokenizer. It is in `preprocessed_data/SNLI/snli_1.0/snli_{train, dev, test}.pkl`.

Preprocessed SST-2 and SNLI is provided in the `preprocessed_data/`. Wikitext-2 is provided in `generative_model/data`.

## Code

For all of these, you might have to edit the Google Drive directory that each notebook mounts to. They should also be run in a GPU-accelerated Google Colab environment.

### Training
The `{bert, lstm, cnn}-sst2.ipynb` files run through training and saving a model on SST-2. The training and fine-tuning code is adapted from [Chris McCormick's](https://mccormickml.com/) [tutorial](https://mccormickml.com/2019/07/22/BERT-fine-tuning/). 

The `lstm-snli.ipynb` file will train a Bi-LSTM on the SNLI dataset.

### Input-Marginalization (Evaluation)
In order to get the figures that are seen in the results section we can run these few main files:
`figure2` will reproduce figure 2 in the original paper for {CNN, LSTM, BERT} trained on SST-2.
`snli_input_marge_v2.ipynb` replicates figure 2 for LSTM trained on SNLI.
`figure3` will reproduce figure 3a in the original paper.
`figure4` will reproduce figure 3b in the original paper.

## Models

The models, as well as their training and testing stats are saved, and can be found in this [Google Drive folder](https://drive.google.com/drive/folders/1j7VFnPhvn9Yg3fjx1flCQy3tHZlUs0mi?usp=sharing).

We fine-tuned a BERTForSentenceClassification model from HuggingFace for SST-2.

## Results
**Table 1: Test accuracy of target models**
| Corpus      | LSTM | BERT | CNN  |
| ----------- | ---- | ---- | ---- |
| SST-2       | 0.77 | 0.92 | 0.75 |
| SNLI        | 0.67 |  --  |  --  |

**Table 2: Comparison of AUCrep with existing erasure scheme (lower is better)**
| Zero | Unk | Input-Marginalization |
| ---  | --- | --------------------- |
| 0.5608 | 0.5328 | 0.4834 |

## Dependencies
`PyTorch, HuggingFace, pytorch_pretrained_bert`

## Difficulty
We started by implementing each of the models, were some of the difficulties lay in preprocessing and formatting the data correctly, as well as improving our accuracy. We had some trouble with implementing the input marginalization, and had to adjust our code to work for BERT, as well as our 3 other models. Finally, we ran into some subtle bugs when computing our AUC curve that took time to identify and solve. Overall we are proud of the work we accomplished on this project despite these challenges!
