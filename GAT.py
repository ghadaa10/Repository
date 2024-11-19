# importing libraries

import json
import networkx as nx
import nltk
import numpy as np
import os
import re
import string
import torch
import torch.nn as nn

from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch.optim.lr_scheduler import StepLR
from nltk.stem import WordNetLemmatizer

# setting up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device
bert = SentenceTransformer("all-MiniLM-L6-v2").to(device)  # for embedding
bert.train()

# ____________________________Preprocessing and formatting data_______________________________#

stopwords = nltk.corpus.stopwords.words(
    "english"
)  # these are the words that appear very frequently but which do not bring meaning to the sentence
stopwords = nltk.download("stopwords")
words = set(nltk.corpus.words.words())  # list of english words
words = nltk.download("words")
lemmatizer = (
    WordNetLemmatizer()
)  # to preserve the root of words in such a way that two words having the same stem will be considered as the same word


def remove_angle_brackets(text):
    """this function removes string between '<>'
    Args:
        text (str): a text
    Returns:
        modified_text(str): the text without string between '<>'

    """
    pattern = re.compile(r"<.*?>")

    # Use the sub() function to replace matches with an empty string
    modified_text = re.sub(pattern, "", text)
    return modified_text


def Preprocess_listofSentence(listofSentence):
    """
    This function preprocesses the text
    Args:
        listofSentence(list): a list of string
    Returns:
        preprocess_list(list): list of the preprocessed text
    """
    preprocess_list = []
    for sentence in listofSentence:
        sentence = remove_angle_brackets(sentence)

        sentence_w_punct = "".join(
            [i.lower() for i in sentence if i not in string.punctuation]
        )  # delete ponctuation

        sentence_w_num = "".join(
            i for i in sentence_w_punct if not i.isdigit()
        )  # delete digits

        tokenize_sentence = nltk.tokenize.word_tokenize(
            sentence_w_num
        )  # transform sentences into a list of tokens

        words_w_stopwords = [
            i for i in list(tokenize_sentence) if i not in list(stopwords)
        ]  # delete stopwords

        words_lemmatize = (
            lemmatizer.lemmatize(w) for w in words_w_stopwords
        )  # lemmatize words

        sentence_clean = " ".join(
            w for w in words_lemmatize if w.lower() in words or not w.isalpha()
        )  # remove capital letters

        preprocess_list.append(sentence_clean)

    return preprocess_list


def embedding_sentence(texts):
    """ "
    This function does the embedding of the text
    Args:
        texts(list): list of text
    Return:
        (list[tensor]):encoded code
    """
    preprocess_list = Preprocess_listofSentence(texts)
    return bert.encode(preprocess_list)


def vectorize_attributes(path, files):
    """This function computes the embedding of edge attributes ('Complement', 'Elaboration' ...)
    Args:
        path: the path vers the folder (training or test)
        files: the data (json) files
    Return:
        edge_attribute(dict): dict of embedding of edge attributes
    """
    # compute the given string attributes
    set_of_attributes = (
        set()
    )  # the set of all given string attributes ('Complement', 'Elaboration' ...)
    for file_name in tqdm(files):
        graph_file_path = os.path.join(path, f"{file_name}.txt")
        with open(graph_file_path) as f:
            for line in f.readlines():
                parts = line.split()
                set_of_attributes.add(parts[1])
    # print(len(set_of_attributes))

    # compute the embedding of the attributes
    edge_attribute = (
        {}
    )  # keys represent the string attributes and values represent the embedding using the one hot encoding method
    for i in range(len(list(set_of_attributes))):
        L = np.zeros(len(set_of_attributes))
        L[i] = 1  # L containes the embedding (one hot encoding method)
        edge_attribute[list(set_of_attributes)[i]] = L
    return edge_attribute


def vectorize_speakers(speaker_name):
    """This function encodes speaker names using one hot encoding method
    Args:
        speaker_name(str): the name of the speaker
    Returns:
        the encoding of the name
    """
    if speaker_name == "UI":
        return np.array([0, 0, 0, 1])
    elif speaker_name == "ME":
        return np.array([0, 0, 1, 0])
    elif speaker_name == "ID":
        return np.array([0, 1, 0, 0])
    elif speaker_name == "PM":
        return np.array([1, 0, 0, 0])


def data_extraction(path):
    """ this function extracts the data for the model
    Args:
        path(str): 'training' or 'test'
    """
    # files extraction
    files = [file.split(".")[0] for file in os.listdir(path) if file.endswith(".json")]

    # edge attribute embedding
    edge_attribute = vectorize_attributes(path, files)

    # data of the features and graph
    data = []

    for transcription_id in tqdm(files):
        with open(path + "/" + f"{transcription_id}.json", "r") as file:
            transcription = json.load(file)

        # extract features(X_training or X_test) and speakers
        node_attribute = []  # features
        # node_labels=[] #speakers
        text = []
        for utterance in transcription:
            node_attribute.append(utterance["text"])
            text.append(utterance["text"])
            # node_labels.append(vectorize_speakers(utterance['speaker']))
        # embedding of features
        node_attribute = embedding_sentence(node_attribute)

        length = np.zeros((len(node_attribute), 1))
        nodes_labels = np.zeros((len(node_attribute), 4))
        for i in range(len(node_attribute)):
            nodes_labels[i] = vectorize_speakers(transcription[i]["speaker"])
            length[i] = len(text[i])

        # Concatenating the attributes
        nodes_attr = np.hstack([nodes_labels, node_attribute])

        # extract graph data
        node_1 = []  # list of node in the left of edges
        node_2 = []  # list of node in the right of edges
        graph_data = []
        graph_file_path = os.path.join(path, f"{transcription_id}.txt")
        with open(graph_file_path) as f:
            for line in f.readlines():
                parts = line.split()
                node_1.append(int(parts[0]))
                node_2.append(int(parts[-1]))
                graph_data.append(edge_attribute[parts[1]])
        edges = [node_1, node_2]

        data.append(
            (
                torch.tensor(nodes_attr, dtype=torch.float),
                torch.tensor(edges, dtype=torch.int64),
                torch.tensor(graph_data, dtype=torch.float),
                transcription_id,
            )
        )

    return data


# ____________________________Loading data_______________________________#

with open(f"{'training_labels'}.json", "r") as file:
    labels = json.load(file)
training_data = data_extraction("training")

# ____________________________Model_______________________________#

import torch.nn.functional as F


class GATLSTM(torch.nn.Module):
    def __init__(self):
        super(GATLSTM, self).__init__()

        self.conv1 = GATv2Conv(
            in_channels=388, out_channels=64, dropout=0.2, heads=8, edge_dim=16
        )
        self.conv2 = GATv2Conv(
            in_channels=64 * 8,
            out_channels=128,
            dropout=0.2,
            heads=8,
            edge_dim=16,
            concat=False,
        )

        self.fc = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(128, 32)

    def forward(self, data):
        x = data[0]
        edge_index = data[1]
        edge_attr = data[2]

        x = x.to(torch.float)

        x = self.dropout(x)

        h = self.conv1(x, edge_index=edge_index, edge_attr=edge_attr)
        h = F.tanh(h)

        h = self.conv2(h, edge_index=edge_index, edge_attr=edge_attr)
        h = F.tanh(h)

        x = h.unsqueeze(0)
        output, (hidden, cell) = self.lstm(x)
        out = hidden.view(-1, 32)
        out = self.dropout(out)
        out = self.fc(out)
        # out = torch.sigmoid(out)

        return out


model = GATLSTM().to(device)

print(model.parameters)

model = GATLSTM()
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.8 / 0.2]))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
# scheduler = StepLR(optimizer, step_size=1, gamma=0.01)


def train(data, labels_training):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out.reshape((-1)), labels_training)
    loss.backward()
    optimizer.step()


for epoch in tqdm(range(100)):
    for i in range(len(training_data)):
        train(
            training_data[i],
            torch.tensor(labels[training_data[i][3]], dtype=torch.float32),
        )

# ____________________________Predicting_______________________________#

test_data = data_extraction("test")
test_labels = dict()


def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]


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
for i in range(len(test_data)):
    y_pred = model(test_data[i])
    y_pred = (y_pred > 0.5).reshape((-1)).to(torch.int)
    test_labels.__setitem__(test_set[i], y_pred.tolist())
print(test_labels)

with open("test_labels_gat.json", "w") as file:
    json.dump(test_labels, file, indent=4)

print("Saved results in test_labels_gat.json")
