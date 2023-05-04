# Ref: https://stackoverflow.com/questions/42352841/how-to-update-an-existing-conda-environment-with-a-yml-file
# Ref: We used code from assignment 4 as an outline
# Ref: https://github.com/pytorch/opacus/blob/main/examples/mnist.py

# To update conda environment: conda env update --file environment.yaml --prune

# Run this file while in the "tasks" directory using: python run_model.py
# To run a specific model_type, execute any of the following:
# 1) python run_model.py dp-sgd
# 2) python run_model.py base
# 3) python run_model.py (will run base by default)

import sys
import torch
import torch.optim as optim
import torch.nn as nn
import evaluate as e  

from tqdm import tqdm  # Tqdm progress bar
from torch.utils.data import random_split
from transformers import AutoTokenizer
from opacus import PrivacyEngine

import datetime
import time
import os
import itertools

from Summarizer import Summarizer
from train_utils import dataloader, plot_curves, save_results


def train(model, dataloader, optimizer, criterion, max_grad_norm = 0.5, scheduler=None, device='cpu'):
    model.train()

    # Create scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Record total loss
    total_loss = 0.

    # Get the progress bar for later modification
    progress_bar = tqdm(dataloader, ascii=True)
 
    # Mini-batch training
    for batch_idx, data in enumerate(progress_bar):     
        source = data[:,0].transpose(1, 0).to(device)
        target = data[:,1].transpose(1, 0).to(device)

        optimizer.zero_grad()
  
        # Use autocast to enable mixed precision training
        with torch.cuda.amp.autocast():
            summary = model(source, use_checkpointing=True)
            summary = summary.reshape(-1, summary.shape[-1]).to(device)
            target = target.reshape(-1).to(device)
            loss = criterion(summary, target)
 
        scaled_loss = scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()
        progress_bar.set_description_str("Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, rouge, device='cpu'):
    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    progress_bar = tqdm(dataloader, ascii=True)

    with torch.no_grad():
        total_loss = 0.0
        total_rouge = 0.0

        for batch_idx, data in enumerate(progress_bar):
            source = data[:,0].transpose(1, 0).to(device)
            target_orig = data[:,1].transpose(1, 0).to(device)
            
            summary_orig = model(source, use_checkpointing=False).to(device)
            summary = summary_orig.reshape(-1, summary_orig.shape[-1])
            target = target_orig.reshape(-1)

            loss = criterion(summary, target)
            total_loss += loss.item()

            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            decoded_summaries = []
            decoded_targets = []

            summaries = summary_orig.argmax(dim=-1).squeeze().transpose(0,1).tolist()
            targets = target_orig.squeeze(dim=0).transpose(0,1).tolist()
            for i in range(len(summaries)): 
                summary_text = tokenizer.decode(summaries[i], skip_special_tokens=True)
                target_text = tokenizer.decode(targets[i], skip_special_tokens=True)

                decoded_summaries.append(summary_text)
                decoded_targets.append(target_text)

            rouge_result = rouge.compute(predictions=decoded_summaries, references=decoded_targets)
            total_rouge += rouge_result['rouge1']

            progress_bar.set_description_str("Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    avg_rouge = total_rouge / len(dataloader)
    return total_loss, avg_loss, avg_rouge


def main():
    # Run code based on specified model_type from terminal: base (default), dp-sgd
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        if model_type!="base" and model_type!="dp-sgd":
            print("Invalid model_type. Must be one of base or dp-sgd, or leave blank for base.\nExiting program.")
            sys.exit()
    else:
        model_type = "base"
    print(f"Running training for {model_type}!")
    
    input_data = torch.load('../data/processed/tokenized_input_data.pt')
    target_data = torch.load('../data/processed/tokenized_target_data.pt')
    data = torch.cat((input_data, target_data), dim=1)
    train_data, val_data, test_data = random_split(data, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
    # Define hyperparameters for grid search
    learning_rates = [1e-3]
    num_heads_list = [8, 4, 2]
    hidden_sizes = [640, 512, 480]
    
    EPOCHS = 10
    input_size = torch.max(input_data).item() + 1
    batch_size = 128
    output_size = input_size
    max_length = input_data.shape[2]
    pad_token_id = 0
    dropout = 0.1
    
    results = []
    best_score = 10e8
    best_params = {}
    best_path = ''

    # Define data loaders
    train_loader, val_loader, test_loader = dataloader(train_data, val_data, test_data, batch_size=batch_size)

    # Loop through hyperparameters and train models
    for learning_rate, num_heads, hidden_size in itertools.product(
        learning_rates, num_heads_list, hidden_sizes
    ):
        # Initialize model, model modules, optimizer, and loss function
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", device)
        model = Summarizer(input_size, hidden_size, output_size, device, max_length, num_heads, dropout, model_type).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        rouge = e.load("rouge")
        
        # Attach privacy engine to optimizer if running dp-sgd model
        if model_type=="dp-sgd":
            # Define DP-SGD specific hyperparameters
            noise_multiplier = 1.1 
            max_grad_norm = 0.5
            delta = 1e-5

            # Instantiate PrivacyEngine() and wrap optimizer
            privacy_engine = PrivacyEngine()

            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                # poisson_sampling=False,
                grad_sample_mode = "functorch",
            )

            # Define array to store epsilon history
            epsilon_history = []
        
        # Define arrays to store metrics for plotting
        train_losses = []
        val_losses = []
        train_rouge = []
        val_rouge = []

        # Define data loaders
        train_loader, val_loader, test_loader = dataloader(train_data, val_data, test_data, batch_size=batch_size)
        print("Hyperparameters: learning_rate={}, num_heads={}, hidden_size={}".format(learning_rate, num_heads, hidden_size))
        start_time = time.time()
        for epoch in range(EPOCHS):
            print("-----------------------------------")
            print("Epoch:", epoch + 1)
            print("-----------------------------------")
    
            # Train the model
            train_loss, avg_train_loss = train(model, train_loader, optimizer, criterion, device=device)
    
            # Evaluate on validation set
            val_loss, avg_val_loss, avg_val_rouge = evaluate(model, val_loader, criterion, rouge, device=device)
    
            # Evaluate on training set
            _, _, avg_train_rouge = evaluate(model, train_loader, criterion, rouge, device=device)
    
            # Print metrics
            print("Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))
            print("Training ROUGE: %.4f. Validation ROUGE: %.4f. " % (avg_train_rouge, avg_val_rouge))
    
            # Append metrics to arrays for plotting
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_rouge.append(avg_train_rouge)
            val_rouge.append(avg_val_rouge)
        
        end_time = time.time()
        elapsed_time = round(end_time - start_time)
        print(f"Elapsed time: {elapsed_time} seconds")

        # Evaluate on test set
        test_loss, avg_test_loss, avg_test_rouge = evaluate(model, test_loader, criterion, rouge, device=device)
        print("Test Loss:", avg_test_loss)
        print("Test ROUGE:", avg_test_rouge)

        params = {
	            "learning_rate": learning_rate,
	            "num_heads": num_heads,
	            "hidden_size": hidden_size,
	    }

	    # Save the final model
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        trial_name = f"trial_{timestamp}_lr_{learning_rate}_nh_{num_heads}_hs_{hidden_size}"
        path = f"../results/hyperparamater_tuning/{trial_name}/"
        os.makedirs(path)
        torch.save(model.state_dict(), path + f"model_state_dict.pt")

	    # Save hyperparameters and results
        result = {'epochs': EPOCHS,
	              'learning_rate': learning_rate,
	              'input_size': input_size,
	              'hidden_size': hidden_size,
	              'batch_size': batch_size,
	              'output_size': output_size,
	              'max_length': max_length,
	              'num_heads': num_heads,
	              'dropout': dropout,
	              'train_loss': train_losses[-1],
	              'val_loss': val_losses[-1],
	              'test_loss': avg_test_loss,
	              'train_rouge': train_rouge[-1],
	              'val_rouge': val_rouge[-1],
	              'test_rouge': avg_test_rouge,
	              'curve_train_loss': train_losses,
	              'curve_val_loss': val_losses,
	              'curve_train_rouge': train_rouge,
	              'curve_val_rouge': val_rouge,
	              'elapsed_time_sec': elapsed_time,
                'device': device
	              }
        save_results(result, path+'result.csv')

        score = val_losses[-1] # use validation loss for determining best params 
        
        # Update best score and best params
        if score < best_score:
            best_score = score
            best_params = params
            best_path = path

	    # Plot curves
        plot_curves(train_loss_history=train_losses, train_rouge_history=train_rouge,
	                valid_loss_history=val_losses, valid_rouge_history=val_rouge, path=path)
	  
	                
    print("Best hyperparameters:", best_params)
    print("Best score (Validation Loss):", best_score)
    print("See: ", path)


if __name__ == '__main__':
    main()
