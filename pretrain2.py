
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

#model_config = DistilGPT2Config()

#model = T5ForConditionalGeneration.from_pretrained("t5-small")
config = T5Config(vocab_size=30000)
model = T5ForConditionalGeneration(config=config)

#from huggingface_pytorch_trainer import train_with_huggingface_pytorch_trainer
#train_with_huggingface_pytorch_trainer(tokenized_dataset, model, num_rows_for_train=num_rows_for_train, num_rows_for_val=num_rows_for_val)
# from native_pytorch_trainer import train_in_native_pytorch
# train_in_native_pytorch(tokenized_dataset, model, batch_size, num_rows_for_train=num_rows_for_train,
#                         num_rows_for_val=num_rows_for_val)

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
#
#
# training_args = TrainingArguments(
#     output_dir='data/',          # output directory to where save model checkpoint
#     evaluation_strategy="epoch",
# )
# print("-----------training--------------")
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
# )
# trainer.train()


# from torch.optim import AdamW
# optimizer = AdamW(model.parameters(), lr=5e-5)
#
# from transformers import get_scheduler
#
# num_epochs = 1
# num_training_steps = num_epochs * len(small_train_dataset)
# lr_scheduler = get_scheduler(
#         name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
# )



model.save_pretrained("t5_smallspm/")