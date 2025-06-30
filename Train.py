import random
import numpy as np
from matplotlib import pyplot as plt
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Set divice
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load tokenizer and pre-trained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model = model.to(device)

# training dataset
texts = []
labels = []
texts_pos = []
texts_neg = []

# open files
with open('rt-polarity.pos', 'r', encoding='utf-8') as file:
    # read line by line
    while True:
        line = file.readline().rstrip('\n')
        texts.append(line)
        labels.append(1)
        texts_pos.append(line)
        # if read to the end, break
        if not line:
            break

with open('rt-polarity.neg', 'r', encoding='utf-8') as file:
    # read line by line
    while True:
        line = file.readline().rstrip('\n')
        texts.append(line)
        labels.append(0)
        texts_neg.append(line)
        # if read to the end, breakv
        if not line:
            break

mapping = dict(zip(texts, labels))
# shuffle arr1
random.shuffle(texts)
# regenerate arr2
labels = [mapping[x] for x in texts]

inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
labels = torch.tensor(labels).to(device)

dataset = TensorDataset(input_ids, attention_mask, labels)
train_loader = DataLoader(dataset, batch_size=2)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Train the model
model.train()
loss_data_set = []
for epoch in range(20):  # epochs = 20
    loss_data = []
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss_data.append(loss)
        loss.backward()
        optimizer.step()
    # save the model
    model_name = str(epoch) + '_model.pth'
    torch.save(model, model_name)
    loss_data_set.append(loss_data)

with open('loss_data.txt', 'w') as file:
    for sublist in loss_data_set:
        line = ' '.join([str(item) for item in sublist])
        file.write(line + '\n')
         