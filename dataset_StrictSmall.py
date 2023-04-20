import os
from datasets import load_dataset
import datasets
# from torch.utils.data import Dataset
# import torch

# ------------ Step 1: Dataset&Dataloader
def get_datasets():
    """
    Loads train, dev, test data for babylm strict small dataset and returns the corresponding huggingface DatasetDict
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

    train_data_file_paths = get_dataset_file_paths(train_data_dir, '.train')
    dev_data_file_paths = get_dataset_file_paths(dev_data_dir, '.dev')
    test_data_file_paths = get_dataset_file_paths(test_data_dir, '.test')

    data_files =  {
        'train': train_data_file_paths,
        'validation': dev_data_file_paths,
        'test': test_data_file_paths,
    }
    dataset = load_dataset("text", data_files=data_files)
    return dataset


# Pre-process dataset (tokenize, concatenate lines, batch)
def pre_process_dataset(dataset, tokenizer, max_seq_length, batch_size, num_proc):
    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size,
        remove_columns=["text"],
        desc="Tokenizing the loaded dataset."
    )

    # For the following pre-processing refer to: https://discuss.huggingface.co/t/help-understanding-how-to-build-a-dataset-for-language-as-with-the-old-textdataset/5870/2
    # Main data processing function that will concatenate all texts from
    # our dataset and generate chunks of max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop,
        # you can customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together,
    # so group_texts throws away a remainder for each of those groups of 1,000 texts.
    # You can adjust that batch_size here but a higher value might be slower to preprocess.
    tokenized_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Batching the loaded dataset."
    )

    return tokenized_dataset
