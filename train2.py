import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5Model, RobertaModel, RobertaTokenizer, RobertaConfig, \
    DataCollatorForLanguageModeling, TrainingArguments, Trainer, GPT2Model, GPT2Config, T5ForConditionalGeneration, \
    T5Config, RobertaForMaskedLM

from dataset_StrictSmall import load_datasets_from_dir, pre_process_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate


train_data, dataset = load_datasets_from_dir()

#tokenizer first file
spm.SentencePieceTrainer.train(
    input=train_data,
    model_prefix="spiece",
    vocab_size=30000,
    model_type="unigram",
    train_extremely_large_corpus=True,
)

sp = spm.SentencePieceProcessor()
sp.load('spiece.model')
tokenizer = T5Tokenizer(vocab_file='spiece.model')


print("-----------tokenizing--------------")
batch_size = 16  # Batch size used for DataLoader
max_seq_length = 20  # fixed length of the sequences (i.e. num tokens per entry)
map_batch_size = 1000  # batch size used for dataset.map() function during pre-processing
num_proc = 4
tokenized_dataset = pre_process_dataset(dataset, tokenizer, max_seq_length, map_batch_size, num_proc)

num_rows_for_train = tokenized_dataset['train'].num_rows
num_rows_for_val = tokenized_dataset['validation'].num_rows

small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(num_rows_for_train))
small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(num_rows_for_val))


config = RobertaConfig(vocab_size=30000, decoder_start_token_id=tokenizer.pad_token_id)
model = RobertaForMaskedLM(config=config)

# Data Collector
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

metric = evaluate.load("accuracy")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


training_args = TrainingArguments(
    output_dir="results2",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    save_total_limit=10
)


# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()
trainer.save_model("robertaresult")
trainer.save_state()
evaluation_results = trainer.evaluate(eval_dataset=small_eval_dataset)
print(evaluation_results)
