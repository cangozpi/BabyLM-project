from torch.utils.data import DataLoader
import torch


# Refer to: https://huggingface.co/docs/transformers/training#train
# Train in Native PyTorch
def train_in_native_pytorch(tokenized_dataset, model, batch_size):
    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42)#.select(range(1)) 
    small_validation_dataset = tokenized_dataset["validation"].shuffle(seed=42)#.select(range(2))
    # small_test_dataset = tokenized_dataset["test"].shuffle(seed=42)#.select(range(2))

    # Convert to PyTorch Datasets and batch them
    small_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    small_validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # small_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataset = DataLoader(small_train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = DataLoader(small_validation_dataset, batch_size=batch_size, shuffle=False)
    # test_dataset = DataLoader(small_test_dataset, batch_size=batch_size, shuffle=False)
    # Note that at this stage:
    #  batch = {
    #   'input_ids': torch.Size([batch_size, max_seq_len])                                            
    #   'attention_mask': torch.Size([batch_size, max_seq_len])
    #   'labels': torch.Size([batch_size, max_seq_len])
    # }


    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=5e-5)

    from transformers import get_scheduler
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataset)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )


    # Train Model
    from tqdm.auto import tqdm
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        losses = []
        for batch in train_dataset:
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            losses.append(loss.detach().cpu().item())
        import numpy as np
        print(f'Training| epoch: {epoch}, loss: {np.mean(losses)}')
    

    # Evaluate Model
    # import evaluate
    # metric = evaluate.load("accuracy")
    # model.eval()
    # for batch in validation_dataset:
    #     # batch = {k: v.to(device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         outputs = model(**batch)

    #     logits = outputs.logits
    #     predictions = torch.argmax(logits, dim=-1)
    #     metric.add_batch(predictions=predictions, references=batch["labels"])

    # metric.comute()

