from dataset_StrictSmall import load_datasets_from_dir, pre_process_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -------- Step 2: Load provided baseline model and tokenizer
# Tokenizer& petrained model, refer to https://huggingface.co/babylm/t5-base-strict/tree/main
tokenizer = AutoTokenizer.from_pretrained("babylm/t5-base-strict-small")
model = AutoModelForSeq2SeqLM.from_pretrained("babylm/t5-base-strict-small")


# ------------ Step 1: Dataset&Dataloader
# Load train, validation, test data
batch_size = 16 # Batch size used for DataLoader
max_seq_length = 20 # fixed length of the sequences (i.e. num tokens per entry)
map_batch_size = 1000 # batch size used for dataset.map() function during pre-processing
num_proc = 4
dataset = load_datasets_from_dir()
tokenized_dataset = pre_process_dataset(dataset, tokenizer, max_seq_length, map_batch_size, num_proc)


debug = True  # ------------ Step 5: Overfit provided baseline model on a small subset of the training dataset
num_rows_for_train = 2 if debug else tokenized_dataset['train'].num_rows # num_rows from dataset['train'] used for training the model
num_rows_for_val = 2 if debug else tokenized_dataset['validation'].num_rows # num_rows from dataset['validation'] used for validating the model

training_mode = 'native_pytorch_training'
assert training_mode in ['native_pytorch_training', 'huggingface_pytorch_Trainer']


if training_mode == 'huggingface_pytorch_Trainer': # Train with Huggingface Pytorch Trainer
    # ------------ Step 3: Training with Huggingface PyTorch Trainer
    from huggingface_pytorch_trainer import train_with_huggingface_pytorch_trainer
    train_with_huggingface_pytorch_trainer(tokenized_dataset, model, num_rows_for_train=num_rows_for_train, num_rows_for_val=num_rows_for_val)
elif training_mode == 'native_pytorch_training': # Train in Native PyTorch
    # ------------ Step 4: Training in Native Pytorch 
    from native_pytorch_trainer import train_in_native_pytorch
    train_in_native_pytorch(tokenized_dataset, model, batch_size, num_rows_for_train=num_rows_for_train, num_rows_for_val=num_rows_for_val)

print("--------------------------evaluation------------------------------")
from native_pytorch_trainer import evaluate_
evaluate_(tokenized_dataset["test"], model)


