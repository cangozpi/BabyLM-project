# Refer to: https://huggingface.co/docs/transformers/training#train
# Train with Pytorch Trainer
def train_with_huggingface_pytorch_trainer(tokenized_dataset, model, num_rows_for_train, num_rows_for_val):
    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(num_rows_for_train))
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(num_rows_for_val))

    from transformers import TrainingArguments
    training_args = TrainingArguments(output_dir="test_trainer")

    import numpy as np
    import evaluate
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    from transformers import TrainingArguments, Trainer
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        # compute_metrics=compute_metrics,
    )
    trainer.train()