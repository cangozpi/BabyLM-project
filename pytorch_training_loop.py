import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np



def save_checkpoint(ckpt_path, model):
    # Save model and its config
    model.save_model_and_config(ckpt_path)


# Refer to: https://huggingface.co/docs/transformers/training#train
# Train in Native PyTorch
def train_for_num_epochs_in_pytorch_loop(train_dataloader, model, num_epochs, lr=3e-4, grad_norm_clip=1.0, validation_dataloader=None, ckpt_path='save_dir/training_loop_ckpt', logger=None):
    if logger is None:
        class Dummy_Logger:
            def __init__(self, *args, **kwargs):
                pass
            def log_msg_to_console(self, msg):
                pass
            def log_dict_to_file(self, info_dict):
                pass
            def log_to_file(self, entity):
                pass
            def log_scalar_to_tb(self, tag, scalar_value):
                pass
        logger = Dummy_Logger()

    # GPU support
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = model.to(device)
    logger.log_msg_to_console(f'using device: {device}')
    logger.log_to_file(f'using device: {device}')

    optimizer = AdamW(model.parameters(), lr=lr)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Train Model
    # progress_bar = tqdm(range(num_training_steps), desc="Training model")
    model.train()
    loss_list = []
    best_loss = float("inf")
    for epoch in range(num_epochs):
        losses = []
        with tqdm(train_dataloader, unit="batch") as progress_bar:
            for batch in progress_bar:
                # Place data on the correct device
                if device != "cpu":
                    for k, v in batch.items():
                        batch[k] = v.to(device)

                progress_bar.set_description(f"Training: Epoch {epoch}/{num_epochs}")

                outputs = model(**batch) # batch is {'input_ids': ..., 'attention_mask': ..., 'labels': ...}
                #print("outputs: ", outputs)
                loss = outputs.loss
                #loss_list.append(loss)
                #print(loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                optimizer.step()
                lr_scheduler.step()
                # progress_bar.update(1)

                losses.append(loss.detach().cpu().item())
                progress_bar.set_postfix(loss=np.mean(losses))

                # progress_bar.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
        batches_dict = {}
        for i in range(len(model.per_input_losses)):
            inputs = [inputs_.cpu().numpy().tolist() for inputs_ in model.per_input_losses[i][0]]
            inputs = [number for sublist in inputs for number in sublist]
            labels = [label_.item() for label_ in model.per_input_losses[i][1]]
            inputs_labels = {input_val: label_val for input_val, label_val in zip(inputs, labels)}
            batches_dict.update(dict(sorted(inputs_labels.items(), key=lambda x: x[1])))

        print(batches_dict)


        logger.log_scalar_to_tb(tag='Training/Loss (epoch)', scalar_value=np.mean(losses))

        val_losses = []
        if validation_dataloader is not None:
            with tqdm(validation_dataloader, unit="batch") as progress_bar:
                for batch in progress_bar:
                    # Place data on the correct device
                    if device != "cpu":
                        for k, v in batch.items():
                            batch[k] = v.to(device)

                    progress_bar.set_description(f"Validation: Epoch {epoch}/{num_epochs}")

                    with torch.no_grad():
                        outputs = model(**batch) # batch is {'input_ids': ..., 'attention_mask': ..., 'labels': ...}
                    val_loss = outputs.loss


                    val_losses.append(val_loss.detach().cpu().item())
                    progress_bar.set_postfix(loss=np.mean(val_losses))
                    # progress_bar.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
        val_loss = np.mean(val_losses)
        logger.log_scalar_to_tb(tag='Validation/Loss (epoch)', scalar_value=val_loss)
        # Support early stopping, or just save model if validation data is not provided
        if (val_loss < best_loss) or (ckpt_path is None):
            best_loss = val_loss
            save_checkpoint(ckpt_path+f"/{epoch}", model)
            logger.log_msg_to_console(f'Saving model checkpoint to path: {ckpt_path+f"/{epoch}"}, val_loss: {val_loss}')
            logger.log_to_file(f'Saving model checkpoint to path: {ckpt_path+f"/{epoch}"}, val_loss: {val_loss}')

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

# def evaluate_(tokenized_test, model):
#     import evaluate
#     accuracy_metric = evaluate.load("accuracy")
#     precision_metric = evaluate.load("precision")
#     recall_metric = evaluate.load("recall")
#     bleu = evaluate.load("bleu")
#     f1_metric = evaluate.load("f1")
#     model.eval()
#     for batch in tokenized_test:
# #       batch = {k: v.to(device) for k, v in batch.items()}
#         with torch.no_grad():
#              outputs = model(**batch)
#              logits = outputs.logits
#              predictions = torch.argmax(logits, dim=-1)
#              accuracy_result = accuracy_metric.compute(predictions=predictions, references=batch["labels"])
#              precision_result = precision_metric.compute(predictions=predictions, references=batch["labels"])
#              recall_result = recall_metric.compute(predictions=predictions, references=batch["labels"])
#              bleu_result = bleu.compute(predictions=predictions, refrences=batch["labels"])
#              f1_result = f1_metric.compute(predictions=predictions, refrences=batch["labels"])

#         print("accuracy: {}, precision {}, recall {}, bleu {}, f1 {}".format(accuracy_result, precision_result, recall_result, bleu_result, f1_result))




# Test model in Native PyTorch
def test_model(test_dataloader, model, ckpt_path='save_dir/testing', logger=None):
    if logger is None:
        class Dummy_Logger:
            def __init__(self, *args, **kwargs):
                pass
            def log_msg_to_console(self, msg):
                pass
            def log_dict_to_file(self, info_dict):
                pass
            def log_to_file(self, entity):
                pass
            def log_scalar_to_tb(self, tag, scalar_value):
                pass
        logger = Dummy_Logger()

    # GPU support
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = model.to(device)
    logger.log_msg_to_console(f'using device: {device}')
    logger.log_to_file(f'using device: {device}')


    # Test Model
    model.eval()
    best_loss = float("inf")
    losses = []
    with tqdm(test_dataloader, unit="batch") as progress_bar:
        for batch in progress_bar:
            # Place data on the correct device
            if device != "cpu":
                for k, v in batch.items():
                    batch[k] = v.to(device)

            outputs = model(**batch) # batch is {'input_ids': ..., 'attention_mask': ..., 'labels': ...}
            loss = outputs.loss

            losses.append(loss.detach().cpu().item())
            progress_bar.set_postfix(loss=np.mean(losses))
    logger.log_scalar_to_tb(tag='Testing/Loss', scalar_value=np.mean(losses))
    logger.log_msg_to_console(f'Testing/Loss: {np.mean(losses)}')
    logger.log_to_file(f'Testing/Loss: {np.mean(losses)}')

    return loss

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


# def evaluate_(tokenized_test, model):
#     import evaluate
#     accuracy_metric = evaluate.load("accuracy")
#     precision_metric = evaluate.load("precision")
#     recall_metric = evaluate.load("recall")
#     bleu = evaluate.load("bleu")
#     f1_metric = evaluate.load("f1")
#     model.eval()
#     for batch in tokenized_test:
# #       batch = {k: v.to(device) for k, v in batch.items()}
#         with torch.no_grad():
#              outputs = model(**batch)
#              logits = outputs.logits
#              predictions = torch.argmax(logits, dim=-1)
#              accuracy_result = accuracy_metric.compute(predictions=predictions, references=batch["labels"])
#              precision_result = precision_metric.compute(predictions=predictions, references=batch["labels"])
#              recall_result = recall_metric.compute(predictions=predictions, references=batch["labels"])
#              bleu_result = bleu.compute(predictions=predictions, refrences=batch["labels"])
#              f1_result = f1_metric.compute(predictions=predictions, refrences=batch["labels"])

#         print("accuracy: {}, precision {}, recall {}, bleu {}, f1 {}".format(accuracy_result, precision_result, recall_result, bleu_result, f1_result))

