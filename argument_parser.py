import argparse
from pretraining_datasets import load_datasets_from_dir
from tokenizer import train_tokenizer_on_corpus, save_tokenizer_to_path
from pytorch_training_loop import train_for_num_epochs_in_pytorch_loop
from tokenizer import load_tokenizer_from_path
from transformers import AutoConfig
from custom_models import CustomPreTrainingTransformerModel
from pretraining_datasets import get_DataLoaders
from huggingface_trainer_loop import train_for_num_epochs_in_huggingface_trainer


parser = argparse.ArgumentParser(description="Train and load tokenizers, transformer models.")

parser.add_argument("-f", "--flip", metavar="IMAGE_FLIP", help = "Path to your input image")
parser.add_argument("-i", "--image", metavar="IMAGE", default="001.jpg", help = "Path to your input image")

# Add tokenizer arguments:
# ------------------------- Tokenizer Training Parameters:
# length = 1000# 100_000 # num rows from dataset to process while training the tokenizer
# vocab_size = 12_500 # 50257, newly trained tokenizer's target vocab size
# batch_size = 128 # batch_size parameter of tokenizer.train_new_from_iterator() function
# dataset_names = load_datasets_from_dir(dataset_names=None, streaming=False)['train']
# tokenizer_save_path = "./save_dir/saved_tokenizer"
# tokenizer_model_or_path = "gpt2" # 124M parameters, smallest version of GPT2
# -------------------------------------
# Train and save tokenizer functionality arguments:
parser.add_argument("-ts_tok", "--train_and_save_tokenizer", default=False, help = "If True, a tokenizer will be trained and saved according to the arguments passed in.")
parser.add_argument("-l", "--length", type=int, help = "num rows from dataset to process while training the tokenizer.")
parser.add_argument("-vs", "--vocab_size", type=int, help = "newly trained tokenizer's target vocabulary size.")
parser.add_argument("-tok_bs", "--tokenizer_batch_size", type=int, help = "batch_size parameter of tokenizer.train_new_from_iterator() function.")
parser.add_argument('--train_dataset_file_names', nargs='*', default=None, type=str, help = "names of the dataset files to load into training dataset.")
parser.add_argument('--validation_dataset_file_names', nargs='*', default=None, type=str, help = "names of the dataset files to load into validation dataset.")
parser.add_argument('--test_dataset_file_names', nargs='*', default=None, type=str, help = "names of the dataset files to load into test dataset.")
parser.add_argument("-tok_pth", "--tokenizer_save_or_load_path", default="./save_dir/saved_tokenizer", help = "path to save/load the trained tokenizer.")
parser.add_argument("-tok_mdl", "--tokenizer_model", default="gpt2", help = "tokenizer model or path to load.")

# Add create model, load tokenizer, train and save model arguments:
# ------------------------- Create model, load tokenizer, train and save model Parameters:
# torch_training = True # whether to train the model using torch training loop or huggingface Trainer
# train_dataset_names = { # Specify which files to use during training (useful for curriculum learning)
#     'train': ['aochildes.train', 'bnc_spoken.train'],
#     'validation': ['aochildes.dev', 'bnc_spoken.dev'],
#     'test': ['aochildes.test', 'bnc_spoken.test']
# }
# pretraining_task = 'clm' # specifies pretraining objective (e.g. ['clm', 'mlm'])
# model_name = "gpt2" # specifies which model to create (e.g. ['gpt2', 'bert-base-uncased'])
# tokenizer_model_or_path = "gpt2" # specifies which tokenizer to create (e.g. ['gpt2', 'bert-base-uncased'])
# num_epochs = 3 # number of epochs to train the model
# batch_size = 2 # dataloader's batch size
# num_workers = 0 # dataloader's num_workers 
# lr = 3e-4
# grad_norm_clip = 1.0
# ckpt_path = 'save_dir/training_loop_ckpt' # path to save the model checkpoints during training
# =========================================
parser.add_argument("--create_model_load_tokenizer_train_and_save_model", default=False, help = "If True, a model is created according to passed in values, tokenizer is loaded, model is trained and then saved.")
parser.add_argument("-pt_train", "--torch_training", default=True, help = "Whether to train the model using custom torch training loop. If False, uses HuggingFace Trainer.")
parser.add_argument("-trans_mdl", "--transformer_model_name", default="gpt2", help = "Specifies which model to create (e.g. ['gpt2', 'bert-base-uncased']).")
parser.add_argument("-pre_t", "--pretraining_task", default="clm", help = "Specifies the pretraining objective used for training (e.g. ['clm', 'mlm'], 'clm' for causal language modeling, and 'mlm' for masked language modeling objective.).")
parser.add_argument("-e", "--num_epochs", type=int, help = "Number of epochs to train the model.")
parser.add_argument("-train_bs", "--training_batch_size", type=int, help = "DataLoader's batch size.")
parser.add_argument("-nw", "--num_workers", type=int, default=0, help = "DataLoader's num_workers parameter.")
parser.add_argument("-lr", "--learning_rate", type=str, default="1e-3", help = "Learning Rate used for training the model.")
parser.add_argument("--grad_norm_clip", type=float, default=1.0, help = "Gradient norm clipping parameter.")
parser.add_argument("-model_ckpt_path", "--model_checkpoint_path", type=str, default='./save_dir/training_loop_ckpt', help = "Path to save the model checkpoints during training.")


# Load model, and load tokenizer and train them (curriculum learning support) arguments:
# ------------------------- Loading Model and tokenizer, and training Parameters:
parser.add_argument("--load_model_load_tokenizer_and_train", default=False, help = "If True, model and tokenizer is loaded from path, and trained.")
parser.add_argument("-model_load_path", "--model_load_path", type=str, default='./save_dir/training_loop_ckpt', help = "Path to load the saved model from checkpoints saved during training.")



# Parse the arguments
args = vars(parser.parse_args())

# ================================================== 
# Train and save tokenizer:
if args['train_and_save_tokenizer']:
    # train the tokenizer
    dataset_names = None
    if args['train_dataset_file_names'] is not None:
        assert (args['validation_dataset_file_names'] is not None) and (args['test_dataset_file_names'] is not None), "Even though val and test datasets will not be used to train the tokenizer, they should be supplied with valid entries!"
        dataset_names = { # Specify which files to use during training (useful for curriculum learning)
            'train': args['train_dataset_file_names'],
            'validation': args['validation_dataset_file_names'],
            'test': args['test_dataset_file_names']
        }
    print(f'Using dataset_names: {dataset_names if dataset_names is not None else "Loading all of the available .train, .dev, .test dataset files."}')

    train_dataset = load_datasets_from_dir(dataset_names=dataset_names, streaming=False)['train']
    new_tokenizer = train_tokenizer_on_corpus(args['tokenizer_model'], args['length'], \
        args['vocab_size'], args['training_batch_size'], train_dataset)
    # save the trained tokenizer
    save_tokenizer_to_path(new_tokenizer, args['tokenizer_save_or_load_path'])
    print(f'Successfully trained the tokenizer: {new_tokenizer}, and saved the tokenizer to path: {args["tokenizer_save_or_load_path"]}')


# ================================================== 
# Create model, load tokenizer, and save model after training:
if args['create_model_load_tokenizer_train_and_save_model'] == 'True':
    # Get Datasets to use
    dataset_names = None
    if args['train_dataset_file_names'] is not None:
        assert (args['validation_dataset_file_names'] is not None) and (args['test_dataset_file_names'] is not None), "both train, validation, and test dataset file names must be supplied if any one of them is specified for training!"
        dataset_names = { # Specify which files to use during training (useful for curriculum learning)
            'train': args['train_dataset_file_names'],
            'validation': args['validation_dataset_file_names'],
            'test': args['test_dataset_file_names']
        }

    print(f'Using dataset_names: {dataset_names if dataset_names is not None else "Loading all of the available .train, .dev, .test dataset files."}')
    # Load the saved tokenizer
    tokenizer = load_tokenizer_from_path(args['tokenizer_save_or_load_path'])
    print(f'Loaded tokenizer: {tokenizer}\n', "-"*50)

    # Initialize Model:
    # Create Config for the transformer model
    config = AutoConfig.from_pretrained(args['transformer_model_name'])
    config.num_labels = tokenizer.vocab_size # make model's vocabulary size match the tokenizer's vocab size
    config.vocab_size = tokenizer.vocab_size # make model's vocabulary size match the tokenizer's vocab size
    config.pretraining_task = args['pretraining_task']
    # config.add_pooling_layer = False

    # Initialize Model from the config for the specified pretraining_task
    model = CustomPreTrainingTransformerModel(config)
    # Train the model
    if args['torch_training'] == 'True':  # Train the model using PyTorch Training Loop
        # Get Pretraining DataLoaders
        train_dataloader, validation_dataloader, test_dataloader = get_DataLoaders(dataset_names, tokenizer, task=args['pretraining_task'], batch_size=args['training_batch_size'], num_workers=args['num_workers'], return_small_debug_dataset=False)
        print('-'*50,f'\nDataloaders:\n\ttrain_dataloader.length: {len(train_dataloader)},\n\tvalidation_dataloader.length: {len(validation_dataloader)},\n\ttest_dataloader.length: {len(test_dataloader)}\n','-'*50)

        # Train the model using pytorch training loop
        model = train_for_num_epochs_in_pytorch_loop(train_dataloader, model, args['num_epochs'], float(args['learning_rate']), args['grad_norm_clip'], validation_dataloader, args['model_checkpoint_path'])

    else: # Train the model using HuggingFace Trainer
        train_for_num_epochs_in_huggingface_trainer(dataset_names, model, tokenizer, args['pretraining_task'], args['training_batch_size'], args['num_epochs'], args['model_checkpoint_path']+"/hugginface_trainer", float(args['learning_rate']))




# ================================================== 
# Load model, tokenizer and train:
if args['load_model_load_tokenizer_and_train'] == 'True':
    # Get Datasets to use
    dataset_names = None
    if args['train_dataset_file_names'] is not None:
        assert (args['validation_dataset_file_names'] is not None) and (args['test_dataset_file_names'] is not None), "both train, validation, and test dataset file names must be supplied if any one of them is specified for training!"
        dataset_names = { # Specify which files to use during training (useful for curriculum learning)
            'train': args['train_dataset_file_names'],
            'validation': args['validation_dataset_file_names'],
            'test': args['test_dataset_file_names']
        }

    print(f'Using dataset_names: {dataset_names if dataset_names is not None else "Loading all of the available .train, .dev, .test dataset files."}')
    # Load the saved tokenizer
    tokenizer = load_tokenizer_from_path(args['tokenizer_save_or_load_path'])
    print(f'Loaded tokenizer: {tokenizer}\n', "-"*50)

    # Load Model:
    model = CustomPreTrainingTransformerModel.load_saved_model(model_save_path=args['model_load_path'])
    print(f'Successfully loaded model: {model}\n','-'*50)

    # Train the model
    if args['torch_training'] == 'True':  # Train the model using PyTorch Training Loop
        # Get Pretraining DataLoaders
        train_dataloader, validation_dataloader, test_dataloader = get_DataLoaders(dataset_names, tokenizer, task=args['pretraining_task'], batch_size=args['training_batch_size'], num_workers=args['num_workers'], return_small_debug_dataset=False)
        print('-'*50,f'\nDataloaders:\n\ttrain_dataloader.length: {len(train_dataloader)},\n\tvalidation_dataloader.length: {len(validation_dataloader)},\n\ttest_dataloader.length: {len(test_dataloader)}\n','-'*50)

        # Train the model using pytorch training loop
        model = train_for_num_epochs_in_pytorch_loop(train_dataloader, model, args['num_epochs'], float(args['learning_rate']), args['grad_norm_clip'], validation_dataloader, args['model_checkpoint_path'])

    else: # Train the model using HuggingFace Trainer
        train_for_num_epochs_in_huggingface_trainer(dataset_names, model, tokenizer, args['pretraining_task'], args['training_batch_size'], args['num_epochs'], args['model_checkpoint_path']+"/hugginface_trainer", float(args['learning_rate']))

# TODO: Load Model from path, and Load tokenizer form path.