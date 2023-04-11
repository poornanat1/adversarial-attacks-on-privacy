# References:
# Using encoded_dict and attention masks: https://stackoverflow.com/a/72647621
# HuggingFace Tokenizer: https://huggingface.co/bert-base-uncased
# Encode_plus: https://huggingface.co/docs/transformers/v4.27.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus
# Get tokens from id: https://stackoverflow.com/questions/63607919/tokens-returned-in-transformers-bert-model-from-encode, https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.convert_ids_to_tokens
# Faker for geenerating fake information: https://github.com/joke2k/faker

# Run this file in the tasks folder using: python data_preprocess.py --file_path '/data/raw/mtsamples.csv' >> ../data/logs/data_preprocess.log
# If you don't have the file, this script can download it from Kaggle automatically

import argparse
import time
import csv
import re
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer
import opendatasets as od
import os
from faker import Faker
import random

parser = argparse.ArgumentParser(description='CS7643 Project')
parser.add_argument('--file_path', default='/data/raw/mtsamples.csv')

def download_data(root):
    # Download data from Kaggle (it will ask for your Kaggle credentials to download this dataset. http://bit.ly/kaggle-creds)
    od.download("https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions", data_dir=root+'/data/')
    os.rename(root + '/data/medicaltranscriptions', root+'/data/raw')

def save_tokens_csv(root, filename, tokenized_data, tokenizer):
    # Write the tokenized data to CSV for visual check
    with open(root + filename, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the headers
        csv_writer.writerow(['tokenized_transcription', 'attention_mask'])

        # Write the tokenized data for each row
        for i in range(len(tokenized_data)):
            # Convert the token IDs to tokens
            tokens = tokenizer.convert_ids_to_tokens(tokenized_data[i].flatten())

            # Loop through the tokenized transcriptions and attention masks
            csv_writer.writerow([tokens])

def remove_stopwords_punctuation(text):
    # Example:
    # tmp = 'ALLOWED CONDITION: , Right shoulder sprain and right rotator cuff tear (partial).,CONTESTED CONDITION:,  AC joint arthrosis right aggravation.,DISALLOWED CONDITION: ,'
    # remove_stopwords_punctuation(tmp)
    # 'allowed condition right shoulder sprain right rotator cuff tear partial contested condition ac joint arthrosis right aggravation disallowed condition'

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if not w in stop_words]

    return ' '.join(words)
    

def add_personal_info(fake, input):
    # Ref: https://github.com/joke2k/faker
    # Create fake personal information using Faker
    # Returns a concatenated string
    profile = fake.profile()
    phone_number = fake.phone_number()
    fields = ['Full name: ' + profile['name'], 
            'Address: ' + profile['address'], 
            'Blood type: ' + profile['blood_group'], 
            'Birthdate: ' + profile['birthdate'].strftime("%m/%d/%Y"),
            'Sex: ' + profile['sex'],
            'Email address: ' + profile['mail'],
            'Occupation: ' + profile['job'],
            'Phone Number: ' + phone_number]
    input = ' '.join(fields) + ' ' + input
    return input

def tokenize_data(file_path):
    # Load the BERT tokenizer (https://huggingface.co/bert-base-uncased)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Initialize a faker generator
    fake = Faker(['en', 'en_AU', 'en_US', 'en_CA', 'en_GB', 'en_IE', 'en_IN', 'en_NZ', 'en_PH', 'en_TH'])
    # Set seeds to 123
    Faker.seed(123)
    random.seed(123)

    # Open the data file and perform tokenization
    with open(file_path, 'r') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)

        # Get the index of the 'transcription' column
        headers = next(csv_reader)
        input_idx = headers.index('transcription')
        target_idx = headers.index('description')

        tokenized_input_data = []
        attention_masks = []
        tokenized_target_data = []
        row_num = 0
        unique_rows = set()

        # Loop through each row in the CSV file
        for row in csv_reader:
            row_num +=1

            # Select transcription and description columns
            input = row[input_idx]
            target = row[target_idx]

            # Ensure minimum length
            if len(input) < 40:
                print(f'Row {row_num} does not meet minimum length requirement of 40. Skipping...')
                continue
            # Ensure unique rows
            if input in unique_rows:
                print(f'Row {row_num} is a duplicate. Skipping...')
                continue
            else:
                unique_rows.add(input)

            # Add personal information to input with probability 50%
            p=0.5
            if random.random() < p:
                print(f'Adding sensitive information to row {row_num}')
                input = add_personal_info(fake, input)

            # Remove stopwords and punctuation before encoding
            input = remove_stopwords_punctuation(input)
            target = remove_stopwords_punctuation(target)

            # Tokenize row (https://huggingface.co/docs/transformers/v4.27.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus)
            encoded_input = tokenizer.encode_plus(
                                text=input,
                                add_special_tokens = True,           # Add '[CLS]' and '[SEP]'
                                max_length = 1000,                   # Pad & truncate all sequences
                                padding = 'max_length',
                                truncation = True,
                                return_attention_mask = True,        # Construct attention masks
                                return_tensors = 'pt',                # Return PyTorch tensors
                                )
            encoded_target = tokenizer.encode_plus(
                                text=target,
                                add_special_tokens = True,           # Add '[CLS]' and '[SEP]'
                                max_length = 1000,                   # Pad & truncate all sequences
                                padding = 'max_length',
                                truncation = True,
                                return_attention_mask = False,        # Construct attention masks
                                return_tensors = 'pt',                # Return PyTorch tensors
                                )

            # Append the encoded row, attention masks, and target to lists
            tokenized_input_data.append(encoded_input['input_ids'])
            attention_masks.append(encoded_input['attention_mask'])
            tokenized_target_data.append(encoded_target['input_ids'])

    return tokenized_input_data, attention_masks, tokenized_target_data, tokenizer

def main():
    start_time = time.time()

    global args
    args = parser.parse_args()
    file_path = args.file_path
    
    # Create new data directory structure
    root = os.path.dirname(os.getcwd())
    if not os.path.exists(root + '/data'):
        os.makedirs(root + '/data')
    if not os.path.exists(root + '/data/processed'):
        os.makedirs(root + 'data/processed')

    # Download data
    file_path = root + file_path
    if not os.path.exists(file_path):
        download_data(root)

    # Tokenize and deduplicate data
    tokenized_input_data, attention_masks, tokenized_target_data, tokenizer = tokenize_data(file_path)

    # Convert the lists to PyTorch tensors
    tokenized_input_tensor = torch.stack(tokenized_input_data, dim=0)
    attention_masks_tensor = torch.stack(attention_masks, dim=0)
    tokenized_target_tensor = torch.stack(tokenized_target_data, dim=0)

    # Save the tensors to disk
    print('Saving token tensors to disk')
    torch.save(tokenized_input_tensor, root + '/data/processed/tokenized_input_data.pt')
    # torch.save(attention_masks_tensor, root + '/data/processed/attention_masks.pt')
    torch.save(tokenized_target_tensor, root + '/data/processed/tokenized_target_data.pt')

    # Save tokens to CSV
    print('Saving tokens as CSV')
    save_tokens_csv(root, '/data/processed/tokenized_input_data.csv', tokenized_input_data, tokenizer)
    save_tokens_csv(root, '/data/processed/tokenized_target_data.csv', tokenized_target_data, tokenizer)

    # Print the shapes of the tensors
    print('Tokenized input shape:', tokenized_input_tensor.shape)
    print('Tokenized target shape:', tokenized_target_tensor.shape)
    print('Attention masks shape:', attention_masks_tensor.shape)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time = round(elapsed_time,1)
    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == '__main__':
    main()