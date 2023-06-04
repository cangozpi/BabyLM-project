# from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import T5Tokenizer, T5Model, RobertaModel, RobertaTokenizer, RobertaConfig, \
    DataCollatorForLanguageModeling, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from dataset_StrictSmall import load_datasets_from_dir
from dataset_StrictSmall import load_datasets_from_dir, load_dataset, pre_process_dataset
from dataset_StrictSmall import load_datasets_from_dir, pre_process_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tokenizers.trainers import BpeTrainer
from transformers import RobertaForMaskedLM
from transformers import AdamW
from transformers import get_scheduler
from native_pytorch_trainer import train_in_native_pytorch


tokenizer = RobertaTokenizer.from_pretrained("data/")
batch_size = 16  # Batch size used for DataLoader
max_seq_length = 20  # fixed length of the sequences (i.e. num tokens per entry)
map_batch_size = 1000  # batch size used for dataset.map() function during pre-processing
num_proc = 4

train_data, dataset = load_datasets_from_dir()
print("-----------tokenizing--------------")
tokenized_dataset = pre_process_dataset(dataset, tokenizer, max_seq_length, map_batch_size, num_proc)
train_dataset = tokenized_dataset['train']
test_dataset = tokenized_dataset['test']
print(len(train_dataset))
model_config = RobertaConfig(vocab_size=30522, max_position_embeddings=max_seq_length)

model = RobertaModel(config=model_config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)


training_args = TrainingArguments(
    output_dir='data/',          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=10,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=16, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)
print("-----------training--------------")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
trainer.train()
trainer.save_model("data/")
model.save_pretrained("pretrained/")
