# plot_curves() is from DL assignment 1
# save_results(): https://www.scaler.com/topics/how-to-create-a-csv-file-in-python/

import torch
import pickle
import io
import numpy as np
import csv

# Torchtest package
import torchtext
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
from torchtext.utils import download_from_url, extract_archive
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt

# Define functions
def dataloader(train_data, val_data, test_data, batch_size):
    train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=False)
    valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

def plot_curves(train_loss_history, train_rouge_history, valid_loss_history, valid_rouge_history, path):
    # learning curves of training and validation loss
    x = list(range(len(train_loss_history)))
    fig_loss = plt.figure()
    ax = fig_loss.add_subplot()
    ax.plot(x, train_loss_history, color='blue', label='Training')
    ax.plot(x, valid_loss_history, color='red', label = 'Validation')
    ax.set(title='Loss', ylabel='Loss', xlabel='Epochs')
    # ax.set_ylim(0, 1)
    plt.legend()
    plt.savefig(path + 'plot_loss.png')
    # plt.show()

    # learning curves of training and validation accuracy
    x = list(range(len(train_rouge_history)))
    fig_acc = plt.figure()
    ax = fig_acc.add_subplot()
    ax.plot(x, train_rouge_history, color='blue', label = 'Training')
    ax.plot(x, valid_rouge_history, color='red', label = 'Validation')
    ax.set(title='ROUGE Score', ylabel='ROUGE', xlabel='Epochs')
    # ax.set_ylim(0, 1)
    plt.legend()
    plt.savefig(path + 'plot_rouge.png')
    # plt.show()

def save_results(data, path):
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
            
