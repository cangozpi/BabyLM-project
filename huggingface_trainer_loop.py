from pretraining_datasets import load_datasets_from_dir, tokenize_dataset, get_data_collator 
from transformers import Trainer, TrainingArguments
import evaluate
import numpy as np

def train_for_num_epochs_in_huggingface_trainer(train_dataset_names, model, tokenizer, pretraining_task, batch_size, num_epochs):
    # Get huggingface dataset
    raw_dataset = load_datasets_from_dir(dataset_names=train_dataset_names, streaming=False)
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)

    # Get corresponding collate_fn for the pretraining task
    pretraining_data_collator = get_data_collator(pretraining_task, tokenizer)

    # Get compute_metrics function to evaluate model performance
    compute_metrics = get_compute_metrics_fn()
    print(compute_metrics)

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
        compute_metrics=compute_metrics
    )

    trainer.train()


def get_compute_metrics_fn():
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    return compute_metrics