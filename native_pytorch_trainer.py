from torch.utils.data import DataLoader
import torch


# Refer to: https://huggingface.co/docs/transformers/training#train
# Train in Native PyTorch
def train_in_native_pytorch(tokenized_dataset, model, batch_size):
    # Convert to PyTorch Datasets and batch them
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataset = DataLoader(tokenized_dataset['train'], batch_size=batch_size, shuffle=True)
    validation_dataset = DataLoader(tokenized_dataset['validation'], batch_size=batch_size, shuffle=False)
    test_dataset = DataLoader(tokenized_dataset['test'], batch_size=batch_size, shuffle=False)
    # Note that at this stage
    #  batch = {
    #   'input_ids': torch.Size([batch_size, max_seq_len])                                            
    #   'attention_mask': torch.Size([batch_size, max_seq_len])
    #   'labels': torch.Size([batch_size, max_seq_len])
    # }


    for batch in train_dataset:
        for k in batch.keys():
            print(k, batch[k].shape)
        import sys
        sys.exit()
        # TODO: implement this