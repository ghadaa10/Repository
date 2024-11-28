# Kaggle - INF554

## INF554 project

By Ghada BEN SLAMA and Sakula HYS

## Description

This project is made of two models that can be used in extractive summarization. Each model has its own file (GAT or LSTM). The GAT model is doing better than the LSTM, but the 2 are quite different so we thought it would be relevant to keep both of them.

## Imports

Before running anything, please make sure that the necessary libraries are installed to run the models. They can be found in the *requirement.txt* document.

## Usage

## LSTM

Change *training_docs* in the **LSTM_based_model.py** file to input the training files. The training labels should be put in the folder with the name *training_labels.json*.
The same should be done for *test_set* for the files which result you want to predict.

Run **LSTM.py**.
The predictions will be stored in *test_labels_lstm.json*.

## GAT

The training labels should be put in the folder with the name *training_labels.json*.

All the training dataset should be put in a folder named *training*, the same goes for the testing dataset (which you want to predict), in a folder *test*.

Run all the cells from **GAT.ipynb**.
The predictions will be stored in *test_labels_gat.json*.

## Support

For any remarks : <ghada.ben-slama@polytechnique.edu> and <sakula.hys@polytechnique.edu>
