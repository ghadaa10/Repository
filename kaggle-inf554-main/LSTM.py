# import necessary libraries and setting up

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torchtext
from torchtext.data import get_tokenizer
from collections import Counter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random_state = 42

# loading the data


def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]


# getting the name of all the training files
training_docs = [
    "ES2002",
    "ES2005",
    "ES2006",
    "ES2007",
    "ES2008",
    "ES2009",
    "ES2010",
    "ES2012",
    "ES2013",
    "ES2015",
    "ES2016",
    "IS1000",
    "IS1001",
    "IS1002",
    "IS1003",
    "IS1004",
    "IS1005",
    "IS1006",
    "IS1007",
    "TS3005",
    "TS3008",
    "TS3009",
    "TS3010",
    "TS3011",
    "TS3012",
]
training_docs = flatten([[m_id + s_id for s_id in "abcd"] for m_id in training_docs])
training_docs.remove("IS1002a")
training_docs.remove("IS1005d")
training_docs.remove("TS3012c")

train = pd.DataFrame()

for doc in training_docs:
    with open("training_labels.json", "r") as file:
        train_labels = json.load(file)
    train_sub = pd.read_json("training/" + doc + ".json")
    train_sub["label"] = train_labels[doc]
    train_sub["doc"] = doc
    train = pd.concat([train, train_sub])

test_set = [
    "ES2003",
    "ES2004",
    "ES2011",
    "ES2014",
    "IS1008",
    "IS1009",
    "TS3003",
    "TS3004",
    "TS3006",
    "TS3007",
]
test_set = flatten([[m_id + s_id for s_id in "abcd"] for m_id in test_set])
test = pd.DataFrame()

for doc in test_set:
    test_sub = pd.read_json("test/" + doc + ".json")
    test_sub["doc"] = doc
    test_sub["label"] = [1] * len(test_sub["text"])
    test = pd.concat([test, test_sub])

# split to obtain a train and validation set
train, valid = train_test_split(train, test_size=0.3, random_state=random_state)

# Preprocessing
tokenizer = get_tokenizer("basic_english")

# Build vocabulary
words = []
num_words = 1000

for text in train["text"]:
    tokens = tokenizer(text)
    words.extend(tokens)

top_1k = dict(Counter(words).most_common(1000))

vocab = torchtext.vocab.vocab(top_1k, specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])  # default index


# maximum length of the sentences?
def max_lth(train):
    max = 0
    for text in train["text"]:
        tokens = tokenizer(text)
        if len(tokens) > max:
            max = len(tokens)
    return max


max_len = max(max_lth(train), max_lth(valid)) + 5


# vectorize the sentences
def vectorize_sentences(sentences, max_len):
    vectors = []
    for text in sentences:
        tokens = tokenizer(text)
        v = vocab.forward(tokens)
        if len(v) > max_len:
            v = v[:max_len]
        if len(v) < max_len:  # padding
            tmp = np.full(max_len, vocab["<pad>"])
            # sentences of length 1 and 2 are 'empty' (only the padding)
            if len(v) > 2:
                tmp[0 : len(v)] = v
            v = tmp
        vectors.append(np.array(v))
    return np.array(vectors)


train_X = vectorize_sentences(train["text"], max_len)
test_X = vectorize_sentences(test["text"], max_len)
val_X = vectorize_sentences(valid["text"], max_len)

train_y = np.array(train["label"]).reshape(-1, 1)
test_y = np.array(test["label"]).reshape(-1, 1)
val_y = np.array(valid["label"]).reshape(-1, 1)

# define batch size
batch_size = 64
batch_size_test = len(test_X)


# create tensor datasets
trainset = TensorDataset(
    torch.from_numpy(train_X).to(device), torch.from_numpy(train_y).float().to(device)
)
validset = TensorDataset(
    torch.from_numpy(val_X).to(device), torch.from_numpy(val_y).float().to(device)
)
testset = TensorDataset(
    torch.from_numpy(test_X).to(device), torch.from_numpy(test_y).float().to(device)
)
# testset = TensorDataset(torch.from_numpy(test_X).to(device))

# create dataloaders
train_loader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(validset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(testset, shuffle=False, batch_size=batch_size_test)


# ___________________________Initializing the training_______________________________#

input_dim = num_words + 2  # add 2 for <unk> and <pad> symbols
embedding_dim = 100
hidden_dim = 32
output_dim = 1

criterion = nn.BCELoss()  # Binary Cross Entropy Loss


def train_model(model, optimizer, loss_criterion):
    iter = 0
    num_epochs = 10
    history_train_acc, history_val_acc, history_train_loss, history_val_loss = (
        [],
        [],
        [],
        [],
    )

    for epoch in range(num_epochs):
        for i, (samples, labels) in enumerate(train_loader):
            # Training mode
            model.train()

            # Load samples
            samples = samples.view(-1, max_len).to(device)
            labels = labels.view(-1, 1).to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(samples)

            # Calculate Loss: softmax --> cross entropy loss
            loss = loss_criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 1000 == 0:
                # Get training statistics
                train_loss = loss.data.item()

                # Testing mode
                model.eval()
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for samples, labels in valid_loader:
                    # Load samples
                    samples = samples.view(-1, max_len).to(device)
                    labels = labels.view(-1).to(device)

                    # Forward pass only to get logits/output
                    outputs = model(samples)

                    # Val loss
                    val_loss = criterion(outputs.view(-1, 1), labels.view(-1, 1))

                    predicted = outputs.ge(0.5).view(-1)

                    # Total number of labels
                    total += labels.size(0)

                    # Total correct predictions
                    correct += (
                        (
                            predicted.type(torch.FloatTensor).cpu()
                            == labels.type(torch.FloatTensor)
                        )
                        .sum()
                        .item()
                    )

                accuracy = 100.0 * correct / total

                # Print Loss
                print(
                    "Iter: {} | Train Loss: {} | Val Loss: {} | Val Accuracy: {}".format(
                        iter,
                        train_loss,
                        val_loss.item(),
                        round(accuracy, 2),
                    )
                )

                # Append to history
                history_val_loss.append(val_loss.data.item())
                history_val_acc.append(round(accuracy, 2))
                history_train_loss.append(train_loss)

    return (history_train_acc, history_val_acc, history_train_loss, history_val_loss)


# ___________________________The model_______________________________#


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        """
        vocab_size: (int) size of the vocabulary - required by embeddings
        embed_dim: (int) size of embeddings
        hidden_dim: (int) number of hidden units
        num_class: (int) number of classes
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # enter here your code
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True
        )  # see documentation
        self.fc = nn.Linear(hidden_dim, num_class)
        self.dropout = nn.Dropout(0.6)

    def forward(self, text):
        r"""
        Arguments:
            text: 1-D tensor representing a bag of text tensors
        """
        # ENTER HERE YOUR CODE
        embedded = self.embedding(text)
        # embedded = embedded.view(-1, max_len*embedding_dim)
        output, (hidden, cell) = self.lstm(embedded)  # activation is already in lstm
        out = hidden.view(-1, self.hidden_dim)
        out = self.dropout(out)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out


# ___________________________Prediction function_______________________________#


def predict(model, test_loader):
    predictions = {}

    # Testing mode
    model.eval()
    # Iterate through test dataset
    for sam in test_loader:
        # pb : sample is not correlated to the document
        # fix: either - find a way to get the document name with the index in the sample and test set
        # - or do the model for each document separately in this function

        # get the tensor
        sam = sam[0]

        # Load samples
        sam = sam.view(-1, max_len).to(device)
        # Forward pass only to get logits/output
        outputs = model(sam)
        predicted = outputs.ge(0.5).view(-1)  # Is a tensor

        # Put each prediction in the right place

        for i in range(len(test)):
            document = test["doc"].values[i]  # TODO: check if it works
            if not document in predictions.keys():
                predictions[document] = []
            predictions[document].append(predicted.tolist()[i])

    return predictions


# convert the predictions to 0s and 1s
def convert(predictions):
    test_labels = {}
    for doc in predictions.keys():
        results = predictions[doc]
        test_labels[doc] = [int(results[i]) for i in range(len(results))]
    return test_labels


# ___________________________Training the model and predicting_______________________________#

model = LSTMModel(input_dim, embedding_dim, hidden_dim, output_dim)
model.to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
(train_acc, val_acc, train_loss, val_loss) = train_model(model, optimizer, criterion)

predictions = predict(model, test_loader)

test_labels = convert(predictions)

with open("test_labels_lstm.json", "w") as file:
    json.dump(test_labels, file, indent=4)

print("Saved results in test_labels_lstm.json")
