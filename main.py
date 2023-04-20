from dataset_StrictSmall import load_datasets_from_dir, pre_process_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader


# -------- Step 2: Load provided baseline model and tokenizer
# Tokenizer& petrained model, refer to https://huggingface.co/babylm/t5-base-strict/tree/main
tokenizer = AutoTokenizer.from_pretrained("babylm/t5-base-strict-small")
model = AutoModelForSeq2SeqLM.from_pretrained("babylm/t5-base-strict-small")


# ------------ Step 1: Dataset&Dataloader
# Load train, validation, test data
batch_size = 16 # Batch size used for DataLoader
max_seq_length = 512
map_batch_size = 1000 # batch size used for dataset.map() function during pre-processing
num_proc = 4
dataset = load_datasets_from_dir()
tokenized_dataset = pre_process_dataset(dataset, tokenizer, max_seq_length, map_batch_size, num_proc)

# Convert to PyTorch Datasets and batch them
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
train_dataset = DataLoader(tokenized_dataset['train'], batch_size=batch_size, shuffle=True)
validation_dataset = DataLoader(tokenized_dataset['validation'], batch_size=batch_size, shuffle=False)
test_dataset = DataLoader(tokenized_dataset['test'], batch_size=batch_size, shuffle=False)
# Note that at this stage
#  batch = {
#   'input_ids': torch.Size([batch_size, max_seq_len])                                            
#   'attention_mask': torch.Size([batch_size, max_seq_len])
#   'labels': torch.Size([batch_size, max_seq_len])
# }


# ------------ Step 3: Overfit provided baseline model on the training dataset
for batch in train_dataset:
    for k in batch.keys():
        print(k, batch[k].shape)
    import sys
    sys.exit()
    # TODO: implement this
