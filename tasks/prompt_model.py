from faker import Faker
from transformers import AutoTokenizer
from Summarizer import Summarizer
import torch
import random
from data_preprocess import remove_stopwords_punctuation

# INSTRUCTIONS
# Update prompts & inputs in main() function and run prompt_model.py

#////////

def prompt_model(prompt, trial_name, model_type, device_type):
    model = load_model(trial_name, model_type, device_type)
    # Load the BERT tokenizer (https://huggingface.co/bert-base-uncased)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    prompt_tokenized = tokenize_prompt(prompt, tokenizer)

    model.eval()
    device = device_type
    with torch.no_grad():
        output = model(prompt_tokenized, use_checkpointing=False).to(device)

    output_decoded = decode_output(output, tokenizer)
    return output_decoded

def load_model(trial_name, model_type, device_type):
    path = f"../results/{trial_name}/model_state_dict.pt"
    # model_state_dict = torch.load(path)

    # TODO load these from results.csv instead of hardcoding
    input_size = 29611 # torch.max(input_data).item() + 1
    hidden_size = 128
    output_size = input_size
    max_length = 32 # input_data.shape[2]
    num_heads = 2
    dropout = 0.1

    device = torch.device(device_type)

    model = Summarizer(input_size, hidden_size, output_size, device, max_length, num_heads, dropout, model_type).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model

# def remove_punctuation(text):
#     # Convert to lowercase
#     text = text.lower()
#     # Remove punctuation using regex
#     text = re.sub(r'[^\w\s]', ' ', text)
#     return text

def tokenize_prompt(prompt, tokenizer):
    # Initialize a faker generator
    fake = Faker(['en', 'en_AU', 'en_US', 'en_CA', 'en_GB', 'en_IE', 'en_IN', 'en_NZ', 'en_PH', 'en_TH'])
    # Set seeds to 123
    Faker.seed(123)
    random.seed(123)

    # Remove stopwords and punctuation before encoding
    prompt_no_stopwords = remove_stopwords_punctuation(prompt)

    # Tokenize row (https://huggingface.co/docs/transformers/v4.27.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus)
    encoded_prompt = tokenizer.encode_plus(
        text=prompt_no_stopwords,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=32,  # Pad & truncate all sequences
        padding='max_length',
        truncation=True,
        return_attention_mask=True,  # Construct attention masks
        return_tensors='pt',  # Return PyTorch tensors
    )
    # # For visual inspection
    # tokens = tokenizer.convert_ids_to_tokens(encoded_prompt.flatten())
    tokenized_prompt = torch.transpose(encoded_prompt['input_ids'],0,1)
    return tokenized_prompt

def decode_output(output, tokenizer):
    decoded_output = []
    # output_words = output.argmax(dim=-1).squeeze().transpose(0, 1).tolist()
    output_words = output.argmax(dim=-1)
    output_words = output_words.squeeze()
    # output_words = output_words.transpose(0, 1)
    output_words = output_words.tolist()
    for i in range(len(output_words)):
        output_text = tokenizer.decode(output_words[i], skip_special_tokens=True)
        decoded_output.append(output_text)
    return decoded_output

def decode_output_str(output, tokenizer):
    output_words = output.argmax(dim=-1)
    output_words = output_words.squeeze()
    output_text = tokenizer.decode(output_words, skip_special_tokens=True)
    return output_text

def main():
    prompt = "prostate cancer diagnosis"
    trial_name = "trial_2023-05-02_21-18_lr_0.001_bs_32"
    model_type = "base" # dp-sgd or base
    device_type = 'cpu' # cuda or cpu ; device being loaded on
    output = prompt_model(prompt, trial_name, model_type, device_type)
    print(output)


if __name__ == '__main__':
    main()