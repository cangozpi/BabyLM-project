from transformers import AutoTokenizer, set_seed, pipeline, AutoModelForCausalLM
from dataset_StrictSmall import load_datasets_from_dir
from tqdm.auto import tqdm
import numpy as np

debug = True

set_seed(42)

# Get tokenizer for the model
tokenizer_model_or_path = "gpt2" # 124M parameters, smallest version of GPT2
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_or_path)
print("Old tokenizer:", tokenizer, "\n")
# print(tokenizer("hahahaha yahu ne komik adamsin", return_special_tokens_mask=True))
# print("names of the fields that the model expects in its forward pass:", tokenizer.model_input_names)
# print(tokenizer.bos_token)
# print(tokenizer.eos_token)
# print(tokenizer.eos_token_id)
# print(tokenizer.unk_token)
# text = "Replace me by any text you'd like."
# encoded_text = tokenizer(text)
# tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
# print(encoded_text)
# print(tokens)

# Train the tokenizer on your corpus from scratch
length = 100_000 # num rows from dataset to process while training the tokenizer
vocab_size = 12_500 # 50257, newly trained tokenizer's target voab size
batch_size = 128

# Load dataset and pick lines from it to train on
train_dataset = load_datasets_from_dir(streaming=False)['train']
idx = np.random.choice(np.arange(train_dataset.num_rows), length)
train_dataset = train_dataset.select(idx)

if debug:
    train_dataset = train_dataset.select(range(0, 4000))

def batch_iterator(dataset, batch_size=10):
    for i in tqdm(range(0, (dataset.num_rows - batch_size + 1), batch_size), desc="Training tokenizer"):
        yield dataset['text'][i:i+batch_size]

new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(train_dataset, batch_size=batch_size), vocab_size=vocab_size) # train the tokenizer
print("New tokenizer:", new_tokenizer)

#-------------------------------------------------- 
# TODO: save and load tokenizer funcitonality
#-------------------------------------------------- 


# ------------------ Generating text with the model (LM) -------------------- 
# Approach using pipeline for generating text using the pretrained model
# model = AutoModelForCausalLM.from_pretrained("gpt2")
# generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
# print(generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5))

# Approach using Torch for generating text using the pretrained model
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# model = AutoModelForCausalLM.from_pretrained("gpt2")
# output = model(**encoded_input)
# print(output)