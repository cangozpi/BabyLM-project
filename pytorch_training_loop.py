import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np


# Refer to: https://huggingface.co/docs/transformers/training#train
# Train in Native PyTorch
def train_for_num_epochs_in_pytorch_loop(train_dataloader, model, num_epochs):
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Train Model
    # progress_bar = tqdm(range(num_training_steps), desc="Training model")
    model.train()
    for epoch in range(num_epochs):
        losses = []
        with tqdm(train_dataloader, unit="batch") as progress_bar:
            for batch in progress_bar:
                progress_bar.set_description(f"Epoch {epoch}/{num_epochs}")

                outputs = model(**batch) # batch is {'input_ids': ..., 'attention_mask': ..., 'labels': ...}
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                # progress_bar.update(1)

                losses.append(loss.detach().cpu().item())
                progress_bar.set_postfix(loss=loss.detach().cpu().item())
                # progress_bar.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

            print(f'Training | epoch: {epoch}, loss: {np.mean(losses)}')
    

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

    return model

def evaluate_(tokenized_test, model):
    import evaluate
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    bleu = evaluate.load("bleu")
    f1_metric = evaluate.load("f1")
    model.eval()
    for batch in tokenized_test:
#       batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
             outputs = model(**batch)
             logits = outputs.logits
             predictions = torch.argmax(logits, dim=-1)
             accuracy_result = accuracy_metric.compute(predictions=predictions, references=batch["labels"])
             precision_result = precision_metric.compute(predictions=predictions, references=batch["labels"])
             recall_result = recall_metric.compute(predictions=predictions, references=batch["labels"])
             bleu_result = bleu.compute(predictions=predictions, refrences=batch["labels"])
             f1_result = f1_metric.compute(predictions=predictions, refrences=batch["labels"])

        print("accuracy: {}, precision {}, recall {}, bleu {}, f1 {}".format(accuracy_result, precision_result, recall_result, bleu_result, f1_result))

