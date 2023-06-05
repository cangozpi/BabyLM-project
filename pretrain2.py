
import sentencepiece as spm

from transformers import T5Tokenizer, T5Model, RobertaModel, RobertaTokenizer, RobertaConfig, \
    DataCollatorForLanguageModeling, TrainingArguments, Trainer, GPT2Model, GPT2Config, T5ForConditionalGeneration, \
    T5Config

from dataset_StrictSmall import load_datasets_from_dir, pre_process_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


train_data, dataset = load_datasets_from_dir()
#tokenizer = AutoTokenizer.from_pretrained("t5-small")
batch_size = 32
total_batches = len(dataset['train']) // batch_size

tokenizer = spm.SentencePieceTrainer.train(
    input=train_data,
    model_prefix="spiece",
    vocab_size=30000,
    model_type="unigram",
    train_extremely_large_corpus=True,
)
tokenizer.save("t5_smallspm/")

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


#model_config = DistilGPT2Config()

#model = T5ForConditionalGeneration.from_pretrained("t5-small")
model = T5Model(T5Config)
#tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print(small_train_dataset[0])
training_args = TrainingArguments(
    output_dir='data/',          # output directory to where save model checkpoint
    evaluation_strategy="steps",    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=1,            # number of training epochs, feel free to tweak
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
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
)
trainer.train()
model.save_pretrained("t5_smallspm/")