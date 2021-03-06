# -*- coding: utf-8 -*-
"""input_marge_v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/ronakdm/input-marginalization/blob/main/input_marge_v2.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install pytorch_pretrained_bert
# !pip install transformers
# !git clone https://github.com/ronakdm/input-marginalization.git

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# cd input-marginalization
# git pull
# cd ..

import sys
sys.path.append("input-marginalization")
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertTokenizer, BertModel
from utils import generate_dataloaders
from models import LSTM
from torch.nn import LogSoftmax
import math
import torch.nn.functional as F
import numpy as np
from google.colab import drive
drive.mount('/content/gdrive',force_remount=True)
save_dir = "/content/gdrive/My Drive/input-marginalization"

SAMPLE_SIZE = 3
SIGMA = 1e-4
log_softmax = LogSoftmax(dim=0)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

cnn = torch.load(f"{save_dir}/cnn_sst2.pt")

def loaddata ():
  train_dataloader, validation_dataloader, test_dataloader = generate_dataloaders(1)
  return test_dataloader

def compute_probability(model, input_ids, attention_masks, label):
    logits = model(
        input_ids,  attention_mask=attention_masks, labels=label.repeat((len(input_ids))),
    ).logits
    
    return math.exp(logits[0][label])

def compute_probability2(model, input_ids, attention_masks, label):
  
    logits = model(
        input_ids.to(torch.int64), attention_mask=attention_masks, labels=label.repeat((len(input_ids))),
    ).logits

    return torch.exp(torch.reshape(logits[:, label], (-1,)))

def calculate_woe(model, input_ids, attention_masks, label, sigma):
  device = "cuda" if next(model.parameters()).is_cuda else "cpu"
  bert_model.to(device)
  
  #predictions is the probability distribution of each word in the vocabulary for each word in input sentence
  predictions = bert_model(input_ids)
  predictions = torch.squeeze(predictions)
  predictions = F.softmax(predictions, dim=1)

  #woe is the weight of evidence
  woe = []
  model.eval()

  with torch.no_grad():
    for j in range (len(predictions)):
      word_scores = predictions[j]
      input_batch = input_ids.clone().to(device)
      
      #word_scores_batch calculates the value of the MLM of Bert for each masked word
      #we put 0 for the first input which is unmasked
      word_scores_batch = [0]

      for k in range(len(word_scores)):
        if word_scores[k] > sigma:
           input_batch = torch.cat((input_batch, input_ids), 0)
           input_batch[len(input_batch)-1][j] = k
           word_scores_batch.append(word_scores[k].item())
      
      #probability_input calculates the p(label|sentence) of the target model given each masked input sentence
      probability_input = compute_probability2(model, input_batch, attention_masks, label)
      
      m = torch.dot(torch.tensor(word_scores_batch).to(device), probability_input)
      logodds_input = math.log(probability_input[0] / (1-probability_input[0]))
      logodds_m = math.log(m / (1-m))
      woe.append(logodds_input-logodds_m)
  return woe

def input_marg(model): 
  test_data = loaddata()
  device = "cuda" if next(model.parameters()).is_cuda else "cpu"
  iter_data = iter(test_data)

  for i in range(SAMPLE_SIZE):
    nextsample = next(iter_data)
    inputsequences = nextsample[0].to(device)
    inputmask =  nextsample[1].to(device)
    labels = nextsample[2].to(device)
    print("")
    print(tokenizer.convert_ids_to_tokens(inputsequences[0][1:11]))
    print(calculate_woe(model, torch.unsqueeze(inputsequences[0][1:11],0),torch.unsqueeze(inputmask[0][:11],0),  torch.unsqueeze(labels[0],0), SIGMA))


def calculate_woe_mc(model, input_ids, attention_masks, label, sigma):
  device = "cuda" if next(model.parameters()).is_cuda else "cpu"
  bert_model.to(device)
  
  #predictions is the probability distribution of each word in the vocabulary for each word in input sentence
  predictions = bert_model(input_ids)
  predictions = torch.squeeze(predictions)
  predictions = F.softmax(predictions, dim=1)
  
  #woe is the weight of evidence
  woe = []
  model.eval()
  #ransom sampling n words from BERT MLM distribution
  n = 1000
  with torch.no_grad():
    for j in range (len(predictions)):
      #convert tensor to numpy array
      predictions = predictions.cpu().numpy()
      indicies = np.random.choice(len(predictions[j]),n,p=predictions[j])
      predictions = torch.tensor(predictions).to(device)
    
      word_scores = [predictions[j][i] for i in indicies]
      #normalizing probabilities based on the random sampled words
      word_scores_n = [float(i)/sum(word_scores) for i in word_scores]
      input_batch = input_ids.clone().to(device)
      
      #word_scores_batch calculates the value of the MLM of Bert for each masked word
      #we put 0 for the first input which is unmasked
      word_scores_batch = [0]

      for k in range(len(word_scores_n)):
        if word_scores_n[k] > sigma:
           input_batch = torch.cat((input_batch, input_ids), 0)
           input_batch[len(input_batch)-1][j] = k
           word_scores_batch.append(word_scores_n[k].item())
      
      #probability_input calculates the p(label|sentence) of the target model given each masked input sentence
      probability_input = compute_probability2(model, input_batch, attention_masks, label)     
      m = torch.dot(torch.tensor(word_scores_batch).float().to(device), probability_input.float())
      logodds_input = math.log(probability_input[0] / (1-probability_input[0]))
      logodds_m = math.log(m / (1-m))
      woe.append(logodds_input-logodds_m)
  return woe
