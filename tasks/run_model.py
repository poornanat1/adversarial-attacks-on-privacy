# Run this file while in the "tasks" directory using: python run_model.py
# If you don't have the "evaluate" package for rouge_score in your environment, run: conda env update --file environment.yaml --prune
    #source: https://stackoverflow.com/questions/42352841/how-to-update-an-existing-conda-environment-with-a-yml-file

#Sources: We used code from assignment 4 as an outline

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import evaluate as e #import ROUGE metric #reference: https://huggingface.co/course/chapter7/5?fw=tf#metrics-for-text-summarization
from tqdm import tqdm_notebook, tqdm # Tqdm progress bar
from torch.utils.data import random_split, DataLoader
from Summarizer import Summarizer

# Define functions
def dataloader(train_data, val_data, test_data, batch_size):
    train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=False)#, collate_fn=generate_batch) #TODO delete collate_fn if not needed
    valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader,valid_loader,test_loader

def train(model, dataloader, optimizer, criterion, scheduler=None, device='cpu'):
    model.train()
    # Record total loss
    total_loss = 0.
    # Get the progress bar for later modification
    progress_bar = tqdm(dataloader, ascii=True)

    for batch_idx, data in enumerate(progress_bar):
        source = data[0].transpose(1, 0).to(device)
        target = data[1].transpose(1, 0).to(device)

        optimizer.zero_grad()

        summary = model(source)
        summary = summary.reshape(-1, summary.shape[-1])

        loss = criterion(summary, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_description_str("Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, rouge, device='cpu'):
    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    progress_bar = tqdm_notebook(dataloader, ascii=True)

    with torch.no_grad():
        total_loss = 0.0
        total_rouge = 0.0
        for batch_idx, data in enumerate(progress_bar):
            source = data[0].transpose(1, 0).to(device)
            target = data[1].transpose(1, 0).to(device)

            summary = model(source)
            summary = summary.reshape(-1, summary.shape[-1])

            loss = criterion(summary, target)
            total_loss += loss.item()

            # Calculate ROUGE score
            summary_text = summary.argmax(dim=2).transpose(1, 0) # convert summary back to text
            target_text = target.reshape(-1, target.shape[1]).argmax(dim=1) # convert target back to text
            rouge_result = rouge.compute(predictions=summary_text, references=target_text, use_aggregator=True)
            total_rouge += rouge_result['rogue1']

            progress_bar.set_description_str("Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    avg_rouge = total_rouge / len(dataloader)
    return total_loss, avg_loss, avg_rouge

def main():
    # Load preprocessed data
    data = torch.load('../data/processed/tokenized_data.pt')
    train_data, val_data, test_data = random_split(data, [0.8, 0.1, 0.1])

    # Define data loaders
    batch_size = 128
    train_loader, val_loader, test_loader = dataloader(train_data, val_data, test_data, batch_size=batch_size)

    # Define hyperparameters
    EPOCHS = 2
    learning_rate = 1e-3

    # Initialize model, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    input_size, emb_size, linear_size = 1000, 10, 10
    model = Summarizer(input_size, emb_size, linear_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    rouge = e.load("rouge")
    criterion = nn.CrossEntropyLoss(ignore_index=0) 

    # Define arrays to store metrics for plotting
    train_losses = []
    val_losses = []
    train_rouge = []
    val_rouge = []

    for epoch in range(EPOCHS):
        print("-----------------------------------")
        print("Epoch:", epoch+1)
        print("-----------------------------------")

        # Train the model
        train_loss, avg_train_loss = train(model, train_loader, optimizer, criterion, device=device)

        # Evaluate on validation set
        val_loss, avg_val_loss, avg_val_rouge = evaluate(model, val_loader, criterion, rouge, device=device)

        # Evaluate on training set
        train_loss, avg_train_loss, avg_train_rouge = evaluate(model, train_loader, criterion, rouge, device=device)

        # Print metrics
        print("Training Loss:", avg_train_loss)
        print("Validation Loss:", avg_val_loss)
        print("Training ROUGE:", avg_train_rouge)
        print("Validation ROUGE:", avg_val_rouge)

        # Append metrics to arrays for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_rouge.append(avg_train_rouge)
        val_rouge.append(avg_val_rouge)

    # Evaluate on test set
    test_loss, avg_test_loss, avg_test_rouge = evaluate(model, test_loader, criterion, rouge, device=device)
    print("Test Loss:", avg_test_loss)
    print("Test ROUGE:", avg_test_rouge)

if __name__ == '__main__':
    main()