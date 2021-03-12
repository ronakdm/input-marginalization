import json
from transformers import BertTokenizer
import pickle

fname='snli_1.0_train.jsonl'

with open(fname) as f:
    json_list = list(f)

json_data = []
for json_line in json_list:
    json_data.append(json.loads(json_line))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

s1, s2, l = [],[],[]

for dp in json_data:
    if dp['gold_label'] != '-':
        s1.append(tokenizer(dp['sentence1'])['input_ids'])
        s2.append(tokenizer(dp['sentence2'])['input_ids'])
        l.append(dp['gold_label'])

with open('snli_train.pkl', 'wb') as f:
    pickle.dump({'s1': s1, 's2': s2, 'labels': l}, f)
