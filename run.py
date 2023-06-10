from pretraining_datasets import get_DataLoaders
from pytorch_training_loop import train_for_num_epochs_in_pytorch_loop
from transformers import Trainer, TrainingArguments

# ============================== Parameters:
torch_training = True # whether to train the model using torch training loop or huggingface Trainer
train_dataset_names = { # Specify which files to use during training (useful for curriculum learning)
    'train': ['aochildes.train', 'bnc_spoken.train'],
    'validation': ['aochildes.dev', 'bnc_spoken.dev'],
    'test': ['aochildes.test', 'bnc_spoken.test']
}
pretraining_taks = 'clm' # specifies pretraining objective
num_epochs = 3 # number of epochs to train the model
batch_size = 2 # dataloader's batch size
num_workers = 0 # dataloader's num_workers 
# =========================================



# get a dummy pre-tokenizer # TODO: change this using argparse to laoding a saved tokenizer !
from transformers import AutoTokenizer
tokenizer_model_or_path = "gpt2" # 124M parameters, smallest version of GPT2
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_or_path)


# get a dummy model # TODO: change this using argparse to initializing or loading a saved model !
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Train the model
if torch_training == True: 
    # Get Pretraining DataLoaders
    train_dataloader, validation_dataloader, test_dataloader = get_DataLoaders(train_dataset_names, tokenizer, task=pretraining_taks, batch_size=batch_size, num_workers=num_workers)
    print('-'*50,f'\nDataloaders:\n\ttrain_dataloader.length: {len(train_dataloader)},\n\tvalidation_dataloader.length: {len(validation_dataloader)},\n\ttest_dataloader.length: {len(test_dataloader)}\n','-'*50)

    # Train the model using pytorch training loop
    model = train_for_num_epochs_in_pytorch_loop(train_dataloader, model, num_epochs)
else:
    # Get huggingface dataset
    from pretraining_datasets import load_datasets_from_dir, tokenize_dataset, get_data_collator
    raw_dataset = load_datasets_from_dir(dataset_names=train_dataset_names, streaming=False)
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)

    # Get corresponding collate_fn for the pretraining task
    pretraining_data_collator = get_data_collator(pretraining_taks, tokenizer)




    # Train the model using HuggingFace Trainer
    args = TrainingArguments(
        output_dir="./my_pretraining_output-huggingface-ds",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=num_epochs,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        # fp16=True,
        # push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=pretraining_data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )
    trainer.train()
