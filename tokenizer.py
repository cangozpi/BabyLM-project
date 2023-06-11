from transformers import AutoTokenizer, set_seed, pipeline, AutoModelForCausalLM
from pretraining_datasets import load_datasets_from_dir
from tqdm.auto import tqdm
import numpy as np

def train_tokenizer_on_corpus(tokenizer_model_or_path, length, vocab_size, batch_size, train_dataset, model_max_length):
    """
    Loads in the specified tokenizer and then trains it to have vocabulary size of vocab_size, 
        using length many rows from the specified dataset.
    Inputs:
        length (int):  num rows from dataset to process while training the tokenizer (picked randomly)
        vocab_size (int): newly trained tokenizer's target vocab size
        batch_size (int): batch_size parameter of tokenizer.train_new_from_iterator() function
    Returns:
        tokenizer: tokenizer trained from scratch on the specified corpus
    """
    set_seed(42)
    if train_dataset.num_rows < length:
        raise Exception('length cannot be larger than the rows in the train_dataset!')
    # Load in tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_or_path, model_max_length=model_max_length)

    idx = np.random.choice(np.arange(train_dataset.num_rows), length, replace=False)
    train_dataset = train_dataset.select(idx)

    def batch_iterator(dataset, batch_size=10):
        for i in tqdm(range(0, (dataset.num_rows - batch_size + 1), batch_size), desc="Training tokenizer from scratch on corpus"):
            yield dataset['text'][i:i+batch_size]

    new_tokenizer = old_tokenizer.train_new_from_iterator(batch_iterator(train_dataset, batch_size=batch_size), vocab_size=vocab_size) # train the tokenizer
    return new_tokenizer

def save_tokenizer_to_path(tokenizer, tokenizer_save_path):
    """
    Saves the given tokenizer to the path specified by tokenizer_save_path.
    """ 
    tokenizer.save_pretrained(tokenizer_save_path)


def load_tokenizer_from_path(tokenizer_save_path):
    """
    Loads the tokenizer which is saved at the given path (tokenizer_save_path).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
    return tokenizer




if __name__ == "__main__":
    debug = True
    # ------------------------- Parameters:
    length = 1000# 100_000 # num rows from dataset to process while training the tokenizer
    vocab_size = 12_500 # 50257, newly trained tokenizer's target voab size
    batch_size = 128 # batch_size parameter of tokenizer.train_new_from_iterator() function
    train_dataset = load_datasets_from_dir(dataset_names=None, streaming=False)['train']
    tokenizer_save_path = "./save_dir/saved_tokenizer"
    tokenizer_model_or_path = "gpt2" # 124M parameters, smallest version of GPT2
    # -------------------------------------

    # Get tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_or_path)
    print("Old tokenizer:", tokenizer, "\n", "-"*50)
    # print(tokenizer("just some dummy text", return_special_tokens_mask=True))
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
    if debug:
        train_dataset = train_dataset.select(range(0, 4000))

    new_tokenizer = train_tokenizer_on_corpus(tokenizer_model_or_path, length, vocab_size, batch_size, train_dataset)
    print("New tokenizer:", new_tokenizer, "\n", "-"*50)

    #-------------------------------------------------- 
    # TODO: save and load tokenizer funcitonality
    #-------------------------------------------------- 

    # Checkout save tokenizer
    save_tokenizer_to_path(new_tokenizer, tokenizer_save_path)

    # Checkout load tokenizer
    del tokenizer, new_tokenizer
    loaded_tokenizer = load_tokenizer_from_path(tokenizer_save_path)
    print(f'Loaded tokenizer: {loaded_tokenizer}\n', "-"*50)


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