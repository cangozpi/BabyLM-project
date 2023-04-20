from dataset_StrictSmall import get_datasets, pre_process_dataset 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -------- Step 2: Load provided baseline model and tokenizer
# Tokenizer& petrained model, refer to https://huggingface.co/babylm/t5-base-strict/tree/main
tokenizer = AutoTokenizer.from_pretrained("babylm/t5-base-strict")
model = AutoModelForSeq2SeqLM.from_pretrained("babylm/t5-base-strict")


# ------------ Step 1: Dataset&Dataloader
# Load train, validation, test data
max_seq_length = 512
batch_size = 1000
num_proc = 4
dataset = get_datasets()
tokenized_dataset = pre_process_dataset(dataset, tokenizer, max_seq_length, batch_size, num_proc)

# TODO: Shuffle training data, and batch all data
print(len(tokenized_dataset['train'][0]['input_ids']))


# ------------ Step 3: evaluate pre-trained model with its tokenizer
# TODO: implement this