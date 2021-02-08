# input-marginalization
**Joint work with [Xinyu Ma](https://github.com/maxinY), [Mayura Patwardhan](https://github.com/mayapatward), and [Peter Michael](https://github.com/ptrmcl)**. This repo contains code and data to implement ["Interpretation of NLP models through input marginalization"](https://www.aclweb.org/anthology/2020.emnlp-main.255/) by Kim *et al*.

## Data

One of the datasets used for experimentation is the [Stanford Sentiment Treebank (SST-2)](https://www.kaggle.com/atulanandjha/stanford-sentiment-treebank-v2-sst2). We started with the SST-2 [sentences](https://github.com/frankaging/SST2-Sentence) cleaned by [frankaging](https://github.com/frankaging), which are in the `data` folder. We then preprocessed the data by removing "neutral" sentiment sentences and representing the sentences in the [BERT](https://huggingface.co/transformers/model_doc/bert.html) vocabulary, which can be found in `preprocessed_data`. 

## Code

The `{bert, lstm, cnn}_sst2.ipynb` files run through training and saving a model on SST-2 (you might have to edit the Google Drive directory that it mounts to). They should be run in a GPU-accelerated Google Colab environment. The training and fine-tuning code is adapted from [Chris McCormick's](https://mccormickml.com/) [tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/). 

## Models

The models, as well as their training and testing stats are saved, and can be found in this [Google Drive folder](https://drive.google.com/drive/folders/1j7VFnPhvn9Yg3fjx1flCQy3tHZlUs0mi?usp=sharing).

