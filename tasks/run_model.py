# Run this file while in the "tasks" directory using: python run_model.py
# If you don't have the "evaluate" package for rouge_score in your environment, run: conda env update --file environment.yaml --prune
    #source: https://stackoverflow.com/questions/42352841/how-to-update-an-existing-conda-environment-with-a-yml-file

#Sources: We used code from assignment 4 as an outline

import torch
import torch.optim as optim
import numpy as np
import evaluate as rouge_score #import ROUGE metric #reference: https://huggingface.co/course/chapter7/5?fw=tf#metrics-for-text-summarization
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
    progress_bar = tqdm(dataloader, ascii=True) #Jupyter version: tqdm_notebook(dataloader, ascii=True)

    #TODO implement
    # Mini-batch training
    # for batch_idx, data in enumerate(progress_bar):
    #     source = data[0].transpose(1, 0).to(device)
    #     target = data[1].transpose(1, 0).to(device)
    #
    #     summary = model(source)
    #     summary = summary.reshape(-1, summary.shape[-1])
    #     target = target.reshape(-1)
    #
    #     optimizer.zero_grad()
    #     loss = criterion(summary, target)
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    #     optimizer.step()

        # total_loss += loss.item()
        # progress_bar.set_description_str("Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device='cpu'):
    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    #TODO implement
    # with torch.no_grad():
    #     # Get the progress bar
    #     progress_bar = tqdm_notebook(dataloader, ascii=True)
    #     for batch_idx, data in enumerate(progress_bar):
    #         source = data[0].transpose(1, 0).to(device)
    #         target = data[1].transpose(1, 0).to(device)
    #
    #         summary = model(source)
    #         summary = summary.reshape(-1, summary.shape[-1])
    #         target = target.reshape(-1)
    #
    #         loss = criterion(summary, target)
    #         total_loss += loss.item()
    #         progress_bar.set_description_str(
    #             "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss

def main():
    #load .pt files in as a tensor & combine
    data = torch.load('../data/processed/tokenized_data.pt') #TODO delete
    # input_data = torch.load('../data/processed/tokenized_input_data.pt) #TODO uncomment
    # target_data = torch.load(/data/processed/tokenized_target_data.pt) #TODO uncomment
    # data = torch.concat(description_data) #, summary_data) #TODO check concat dimension and uncomment

    #split data into train, validation, and test
    generator = torch.Generator().manual_seed(42) # define a generator to make results reproducable
    train_data, val_data, test_data = random_split(data, [0.8,0.1, 0.1], generator = generator)

    #define Dataloaders for each train, val, and test
    batch_size = 128
    train_loader, val_loader, test_loader = dataloader(train_data, val_data, test_data, batch_size= batch_size)

    # Define hyperparameters
    EPOCHS = 2 #TODO update
    learning_rate = 1e-3

    # Declare models, optimizer, and loss function
    # set_seed_nb() #TODO uncomment?
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using device: %s" % device)
    input_size, emb_size, linear_size = 1000, 10, 10
    model  = Summarizer(input_size, emb_size, linear_size) #TODO update
    optimizer = optim.Adam(model.parameters(), lr = learning_rate) #TODO update
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer) #TODO update or delete
    criterion = rouge_score.load("rouge") #TODO update to use ROUGE metric https://huggingface.co/course/chapter7/5?fw=tf#metrics-for-text-summarization
    # criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) #TODO delete

    #run training and eval
    # save data for plots:
    transformer_train_perplexity = []
    transformer_val_perplexity = []
    for epoch_idx in range(EPOCHS):
        print("-----------------------------------")
        print("Epoch %d" % (epoch_idx + 1))
        print("-----------------------------------")

        train_loss, avg_train_loss = train(model, train_loader, optimizer, criterion, device=device)
        # scheduler.step(train_loss) TODO uncomment or delete

        val_loss, avg_val_loss = evaluate(model, val_loader, criterion, device=device)

        print("Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))
        print("Training Perplexity: %.4f. Validation Perplexity: %.4f. " % (np.exp(avg_train_loss), np.exp(avg_val_loss)))
        transformer_train_perplexity.append(np.exp(avg_train_loss))
        transformer_val_perplexity.append(np.exp(avg_val_loss))

if __name__ == '__main__':
    main()