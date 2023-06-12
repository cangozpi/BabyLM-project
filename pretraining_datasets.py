import os
from datasets import load_dataset
import datasets
from typing import Dict
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

# ------------ Load train, validation, test datasets:
def load_datasets_from_dir(dataset_names: Dict = None , streaming=False):
    """
    Loads train, dev, test data for babylm strict small dataset and returns the corresponding huggingface DatasetDict
    If dataset_names is None, then all of the available train, dev, test sets are read.
    Returns:
        dataset (DatasetDict)
    """
    def get_dataset_file_paths(data_dir, extension):
        """
        Returns (list of str) file paths of files with matching file extensions (extensions) 
        inside the specified directory (data_dir).
        Inputs:
            data_dir (str)
            extension (str)
        Returns:
            data_file_paths (list of str)
        """
        data_file_paths = []
        for path in os.listdir(data_dir):
            if os.path.isfile(os.path.join(data_dir, path)) and extension in path:
                data_file_paths.append(data_dir + path)
        return data_file_paths

    train_data_dir = './babylm_data/babylm_data/babylm_10M/'
    dev_data_dir = './babylm_data/babylm_data/babylm_dev/'
    test_data_dir = './babylm_data/babylm_data/babylm_test/'

    if dataset_names is None: # load in all the available data files
        train_data_file_paths = get_dataset_file_paths(train_data_dir, '.train')
        dev_data_file_paths = get_dataset_file_paths(dev_data_dir, '.dev')
        test_data_file_paths = get_dataset_file_paths(test_data_dir, '.test')
        data_files =  {
            'train': train_data_file_paths,
            'validation': dev_data_file_paths,
            'test': test_data_file_paths,
        }
    else: # load in specified data files
        data_files = {}
        for split_name, file_names in dataset_names.items():
            if split_name == 'train':
                file_names = list(map(lambda x: train_data_dir + x, file_names))
            if split_name == 'validation':
                file_names = list(map(lambda x: dev_data_dir + x, file_names))
            if split_name == 'test':
                file_names = list(map(lambda x: test_data_dir + x, file_names))

            data_files[split_name] = file_names

    dataset = load_dataset("text", data_files=data_files, streaming=streaming)
    return dataset


# Refer to: https://huggingface.co/learn/nlp-course/chapter7/6#preparing-the-dataset
def tokenize_dataset(raw_dataset, tokenizer):
    """
    Tokenizes the given raw_dataset(huggingface dataset with streaming=False) using the tokenizer and returns the tokenized dataset.
    No padding is done. Inputs are truncated if they are longer than the tokenizer.model_max_length.
    Returned rows are in 'input_ids' and their lengths are <= tokenizer.model_max_length.
    """
    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length > 3:
                input_batch.append(input_ids)
        if len(input_batch) > 0:
            return {"input_ids": input_batch}

    tokenized_dataset = raw_dataset.map(
        tokenize, batched=True, remove_columns=raw_dataset["train"].column_names
    )
    return tokenized_dataset



# ------------ Process the text data (Pad, Mask, add labels) for Causal Language Modeling and Masked Language Modeling pretraining Tasks:

# Refer to: https://huggingface.co/learn/nlp-course/chapter7/6#preparing-the-dataset
def get_data_collator(task, tokenizer, mlm_probability=0.15):
    """
    Returns data collator that corresponds to task(can be causal language modeling or masked language modeling). 
    Returned collator functions pads the sequences in the given batch to the length of the longest sequence (pads right). 
    Adds label and attention_mask columns to the dataset according to the chosen task.
    Note that: shifting the inputs and labels to align them happens inside the model, so the data collator just copies the inputs to create the labels.
    """
    assert task in ['clm', 'mlm']
    # TODO: not sure about using tokenizer.eos_token as pad_token and mask_token as done below !
    if (tokenizer.pad_token is None) and (tokenizer.eos_token is not None):
        tokenizer.pad_token = tokenizer.eos_token # specify pad_token for the tokenizer which will be used for causal LM data collator
    if (tokenizer.mask_token is None) and (tokenizer.eos_token is not None):
        tokenizer.mask_token = tokenizer.eos_token # specify pad_token for the tokenizer which will be used for causal LM data collator
    if task == 'clm':
        # This will pad the inputs to the length of the longest element in the batch, 
        # also creates labels for causal language modeling task as specified by mlm=False
        causal_lm_pretraining_task_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt') 
        return causal_lm_pretraining_task_collator

    elif task == 'mlm':
        # This will pad the inputs to the length of the longest element in the batch, 
        # also creates labels for masked language modeling task as specified by mlm=True
        masked_lm_pretraining_task_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability, return_tensors='pt') 
        return masked_lm_pretraining_task_collator


def get_DataLoaders(train_dataset_names, tokenizer, task='clm', batch_size=32, num_workers=0, return_small_debug_dataset=False):
    """
    Returns 3 dataloaders(train_dataloader, validation_dataloader, test_dataloader) for the chosen pretraining task given the dataset, tokenizer, and other params.
    Inputs:
        train_dataset_names (dict of list): if None then all of the train,dev,test data files will be read into the dataset. If not None, 
            specified as {'train': ['aochildes.train', 'cbc.train'], 'validation': ['aochildes.dev', 'cbc.dev'], 'test': ['aochildes.test', 'cbc.test']}, 
            the specified files are read into dataset splits.
        tokenizer: Huggingface transformer which will be used to tokenize, pad, and mask the dataset.
        batch_size: batch_size parameter of torch DataLoader.
        num_workers: num_workers parameter of torch DataLoader.
    Returns:
        train_dataloader (torch.utils.data.DataLoader): dataloader to be used during pretraining
        validation_dataloader (torch.utils.data.DataLoader): dataloader to be used during pretraining
        test_dataloader (torch.utils.data.DataLoader): dataloader to be used during pretraining
    """
    assert task in ['clm', 'mlm']
    # Load dataset
    raw_dataset = load_datasets_from_dir(dataset_names=train_dataset_names, streaming=False)
    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)
    if return_small_debug_dataset:
        tokenized_dataset['train'] = tokenized_dataset['train'].select(range(40))
        tokenized_dataset['validation'] = tokenized_dataset['validation'].select(range(40))
        tokenized_dataset['test'] = tokenized_dataset['test'].select(range(40))
    # Get data collator function for the chosen pretraining task (Causal LM, Masked LM)
    pretraining_task_data_collator = get_data_collator(task, tokenizer)
    #
    train_dataloader = DataLoader(
        dataset=tokenized_dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pretraining_task_data_collator
    )
    validation_dataloader = DataLoader(
        dataset=tokenized_dataset['validation'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pretraining_task_data_collator
    )
    test_dataloader = DataLoader(
        dataset=tokenized_dataset['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pretraining_task_data_collator
    )
    return train_dataloader, validation_dataloader, test_dataloader




# ------------ Change below as you wish it is for debugging purposes
if __name__ == "__main__":
    # ----------- Check out dataset loading:
    # Specify which files to use during training (useful for curriculum learning)
    train_dataset_names = {
        'train': ['aochildes.train', 'bnc_spoken.train'],
        'validation': ['aochildes.dev', 'bnc_spoken.dev'],
        'test': ['aochildes.test', 'bnc_spoken.test']
    }

    raw_dataset = load_datasets_from_dir(dataset_names=train_dataset_names, streaming=False)
    print(raw_dataset)
    print('-'*50)
    print(raw_dataset['train'])
    print('-'*50)
    print(raw_dataset['train']['text'][:5])
    print('-'*100)

    # ----------- Check out dataset tokenization:
    # get a dummy pre-tokenizer 
    from transformers import AutoTokenizer
    tokenizer_model_or_path = "gpt2" # 124M parameters, smallest version of GPT2
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_or_path)

    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)

    print(tokenized_dataset)
    print('-'*50)
    print(tokenized_dataset['train']['input_ids'][:5])
    print('-'*100)


    # ----------- Check out data_collators:
    causal_lm_pretraining_data_collator = get_data_collator('clm', tokenizer)
    clm_out = causal_lm_pretraining_data_collator([tokenized_dataset["train"][i] for i in range(5)])
    print(clm_out) # Note that: Shifting the inputs and labels to align them happens inside the model, so the data collator just copies the inputs to create the labels.
    print('-'*50)

    masked_lm_pretraining_data_collator = get_data_collator('mlm', tokenizer)
    mlm_out = masked_lm_pretraining_data_collator([tokenized_dataset["train"][i] for i in range(5)])
    print(mlm_out) # Note that: Shifting the inputs and labels to align them happens inside the model, so the data collator just copies the inputs to create the labels.
    print('-'*100)

    
    # ----------- Check out DataLoaders:
    # for Causal Language Model pretraining
    train_dataloader, validation_dataloader, test_dataloader = get_DataLoaders(train_dataset_names, tokenizer, task='clm', batch_size=2, num_workers=0)
    print(f'Causal Language Modeling Dataloaders:\n\ttrain_dataloader.length: {len(train_dataloader)},\n\tvalidation_dataloader.length: {len(validation_dataloader)},\n\ttest_dataloader.length: {len(test_dataloader)}')
    print('-'*50)
    for cur_batch in train_dataloader:
        input_ids, attention_mask, label = cur_batch.input_ids, cur_batch.attention_mask, cur_batch.attention_mask
        print(cur_batch)
        print('-'*50)
        break

    # for Masked Language Model pretraining
    train_dataloader, validation_dataloader, test_dataloader = get_DataLoaders(train_dataset_names, tokenizer, task='mlm', batch_size=2, num_workers=0)
    print(f'Masked Language Modeling Dataloaders:\n\ttrain_dataloader.length: {len(train_dataloader)},\n\tvalidation_dataloader.length: {len(validation_dataloader)},\n\ttest_dataloader.length: {len(test_dataloader)}')
    print('-'*50)
    for cur_batch in train_dataloader:
        input_ids, attention_mask, label = cur_batch.input_ids, cur_batch.attention_mask, cur_batch.attention_mask
        print(cur_batch)
        print('-'*50)
        break
    print('-'*100)