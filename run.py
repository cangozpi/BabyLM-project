from pretraining_datasets import get_DataLoaders
from pytorch_training_loop import train_for_num_epochs_in_pytorch_loop
from huggingface_trainer_loop import train_for_num_epochs_in_huggingface_trainer
from custom_models import CustomPreTrainingTransformerModel
from transformers import AutoConfig

# ============================== Parameters:
torch_training = True # whether to train the model using torch training loop or huggingface Trainer
train_dataset_names = { # Specify which files to use during training (useful for curriculum learning)
    'train': ['aochildes.train', 'bnc_spoken.train'],
    'validation': ['aochildes.dev', 'bnc_spoken.dev'],
    'test': ['aochildes.test', 'bnc_spoken.test']
}
pretraining_task = 'clm' # specifies pretraining objective (e.g. ['clm', 'mlm'])
model_name = "gpt2" # specifies which model to create (e.g. ['gpt2', 'bert-base-uncased'])
tokenizer_model_or_path = "gpt2" # specifies which tokenizer to create (e.g. ['gpt2', 'bert-base-uncased'])
num_epochs = 3 # number of epochs to train the model
batch_size = 2 # dataloader's batch size
num_workers = 0 # dataloader's num_workers 
lr = 3e-4
grad_norm_clip = 1.0
ckpt_path = 'save_dir/training_loop_ckpt'
return_small_debug_dataset = True
# =========================================



# get a dummy pre-tokenizer # TODO: change this using argparse to laoding a saved tokenizer !
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_or_path)


# get a dummy model # TODO: change this using argparse to initializing or loading a saved model !
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("gpt2")

# Initialize Model:
# Create Config for the transformer model
config = AutoConfig.from_pretrained(model_name)
config.num_labels = tokenizer.vocab_size # make model's vocabulary size match the tokenizer's vocab size
config.pretraining_task = pretraining_task
# config.add_pooling_layer = False

# Initialize Model from the config for the specified pretraining_task
model = CustomPreTrainingTransformerModel(config)

# Train the model
if torch_training == True:  # Train the model using PyTorch Training Loop
    # Get Pretraining DataLoaders
    train_dataloader, validation_dataloader, test_dataloader = get_DataLoaders(train_dataset_names, tokenizer, task=pretraining_task, batch_size=batch_size, num_workers=num_workers, return_small_debug_dataset=return_small_debug_dataset)
    print('-'*50,f'\nDataloaders:\n\ttrain_dataloader.length: {len(train_dataloader)},\n\tvalidation_dataloader.length: {len(validation_dataloader)},\n\ttest_dataloader.length: {len(test_dataloader)}\n','-'*50)

    # Train the model using pytorch training loop
    model = train_for_num_epochs_in_pytorch_loop(train_dataloader, model, num_epochs, lr, grad_norm_clip, validation_dataloader, ckpt_path)

else: # Train the model using HuggingFace Trainer
    train_for_num_epochs_in_huggingface_trainer(train_dataset_names, model, tokenizer, pretraining_task, batch_size, num_epochs)