from utils import set_seed, Logger
import argparse
from pretraining_datasets import load_datasets_from_dir
from tokenizer import train_tokenizer_on_corpus, save_tokenizer_to_path
from pytorch_training_loop import train_for_num_epochs_in_pytorch_loop, test_model
from tokenizer import load_tokenizer_from_path
from transformers import AutoConfig
from custom_models import CustomPreTrainingTransformerModel
from pretraining_datasets import get_DataLoaders
from huggingface_trainer_loop import train_for_num_epochs_in_huggingface_trainer


parser = argparse.ArgumentParser(description="Train and load tokenizers, transformer models.")


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
parser.add_argument("--tokenizer_model_max_length", default=1024, type=int, help = "sets tokenizer.model_max_length parameter of the newly created tokenizer. This determines the truncation length during preprocessing the dataset using this tokenizer.")
parser.add_argument('--train_dataset_file_names', nargs='*', default=None, type=str, help = "names of the dataset files to load into training dataset.")
parser.add_argument('--validation_dataset_file_names', nargs='*', default=None, type=str, help = "names of the dataset files to load into validation dataset.")
parser.add_argument('--test_dataset_file_names', nargs='*', default=None, type=str, help = "names of the dataset files to load into test dataset.")
parser.add_argument("-tok_pth", "--tokenizer_save_or_load_path", default="./save_dir/saved_tokenizer", help = "path to save/load the trained tokenizer.")
parser.add_argument("-tok_mdl", "--tokenizer_model", default="gpt2", help = "tokenizer model or path to load.")
parser.add_argument("--seed", type=int, default=42, help = "Seed to set random, numpy, and torch for reproducibility.")

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
parser.add_argument("--hidden_size", type=int, default=768, help = "The hidden size of the model (defaults to 768).")
parser.add_argument("--num_attention_heads", type=int, default=12, help = "The number of attention heads used in the multi-head attention layers of the model (defaults to 12).")
parser.add_argument("--num_hidden_layers", type=int, default=12, help = "The number of blocks in the model (defaults to 12).")


# Load model, and load tokenizer and train them (curriculum learning support) arguments:
# ------------------------- Loading Model and tokenizer, and training Parameters:
parser.add_argument("--load_model_load_tokenizer_and_train", default=False, help = "If True, model and tokenizer is loaded from path, and trained.")
parser.add_argument("-model_load_path", "--model_load_path", type=str, default='./save_dir/training_loop_ckpt', help = "Path to load the saved model from checkpoints saved during training. Also used for logging directory by the Logger.")


# Load model, and load tokenizer and test them arguments:
# ------------------------- Loading Model and tokenizer, and testing Parameters:
parser.add_argument("--load_model_load_tokenizer_and_test", default=False, help = "If True, model and tokenizer is loaded from path, and tested on test dataset.")
parser.add_argument("-test_bs", "--testing_batch_size", type=int, help = "DataLoader's batch size.")

# Parse the arguments
args = vars(parser.parse_args())

# Set seed for reproducibility
set_seed(args['seed'])

# ================================================== 
# Train and save tokenizer:
if args['train_and_save_tokenizer']:
    # Get reference to a Logger
    logger = Logger(log_file_name=args['tokenizer_save_or_load_path']+'/logs/logger.logs', tb_log_dir=None)

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
        args['vocab_size'], args['tokenizer_batch_size'], train_dataset, args['tokenizer_model_max_length'])
    # save the trained tokenizer
    save_tokenizer_to_path(new_tokenizer, args['tokenizer_save_or_load_path'])
    logger.log_msg_to_console(f'Successfully trained the tokenizer: {new_tokenizer}, and saved the tokenizer to path: {args["tokenizer_save_or_load_path"]}')
    logger.log_dict_to_file(args) # log passed in args into a file


# ================================================== 
# Create model, load tokenizer, and save model after training:
if args['create_model_load_tokenizer_train_and_save_model'] == 'True':
    # Get reference to a Logger
    logger = Logger(log_file_name=args['model_checkpoint_path']+'/logs/logger.logs', tb_log_dir=args['model_checkpoint_path']+'/logs/tb_logs')
    logger.log_dict_to_file(args)

    # Get Datasets to use
    dataset_names = None
    if args['train_dataset_file_names'] is not None:
        assert (args['validation_dataset_file_names'] is not None) and (args['test_dataset_file_names'] is not None), "both train, validation, and test dataset file names must be supplied if any one of them is specified for training!"
        dataset_names = { # Specify which files to use during training (useful for curriculum learning)
            'train': args['train_dataset_file_names'],
            'validation': args['validation_dataset_file_names'],
            'test': args['test_dataset_file_names']
        }

    # Load the saved tokenizer
    tokenizer = load_tokenizer_from_path(args['tokenizer_save_or_load_path'])
    logger.log_msg_to_console(f'Loaded tokenizer: {tokenizer}')

    # Initialize Model:
    # Create Config for the transformer model
    config = AutoConfig.from_pretrained(args['transformer_model_name'])
    config.num_labels = tokenizer.vocab_size # make model's vocabulary size match the tokenizer's vocab size
    config.vocab_size = tokenizer.vocab_size # make model's vocabulary size match the tokenizer's vocab size
    config.pretraining_task = args['pretraining_task']
    # Setting Autoconfig parameters
    # Refer to: https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/configuration#transformers.PretrainedConfig ,
    # https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config , and https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig
    if config.model_type == 'gpt2': # GPT2Config overrides these parameters of PretrainedConfig so we handle it specially
        config.n_embd = args['hidden_size'] # The hidden size of the model (defaults to 768)
        config.n_head = args['num_attention_heads'] # The number of attention heads used in the multi-head attention layers of the model (defaults to 12)
        config.n_layer = args['num_hidden_layers'] #  The number of blocks in the model (defaults to 12)
    else:
        config.hidden_size = args['hidden_size'] # The hidden size of the model (defaults to 768)
        config.num_attention_heads = args['num_attention_heads'] # The number of attention heads used in the multi-head attention layers of the model (defaults to 12)
        config.num_hidden_layers = args['num_hidden_layers'] #  The number of blocks in the model (defaults to 12)
    # config.add_pooling_layer = False

    logger.log_to_file(config)
    # Initialize Model from the config for the specified pretraining_task
    model = CustomPreTrainingTransformerModel(config)
    logger.log_to_file(model)
    # Train the model
    if args['torch_training'] == 'True':  # Train the model using PyTorch Training Loop
        # Get Pretraining DataLoaders
        train_dataloader, validation_dataloader, test_dataloader = get_DataLoaders(dataset_names, tokenizer, task=args['pretraining_task'], batch_size=args['training_batch_size'], num_workers=args['num_workers'], return_small_debug_dataset=False)
        logger.log_msg_to_console(f'\nDataloaders:\n\ttrain_dataloader.length: {len(train_dataloader)},\n\tvalidation_dataloader.length: {len(validation_dataloader)},\n\ttest_dataloader.length: {len(test_dataloader)}')
        logger.log_to_file(f'\nDataloaders:\n\ttrain_dataloader.length: {len(train_dataloader)},\n\tvalidation_dataloader.length: {len(validation_dataloader)},\n\ttest_dataloader.length: {len(test_dataloader)}')

        # Train the model using pytorch training loop
        model = train_for_num_epochs_in_pytorch_loop(train_dataloader, model, args['num_epochs'], float(args['learning_rate']), args['grad_norm_clip'], validation_dataloader, args['model_checkpoint_path'], logger=logger)

    else: # Train the model using HuggingFace Trainer
        train_for_num_epochs_in_huggingface_trainer(dataset_names, model, tokenizer, args['pretraining_task'], args['training_batch_size'], args['num_epochs'], args['model_checkpoint_path']+"/hugginface_trainer", float(args['learning_rate']))

    logger.log_to_file('Training finished.')
    logger.log_msg_to_console('Training finished.')


# ================================================== 
# Load model, tokenizer and train:
if args['load_model_load_tokenizer_and_train'] == 'True':
    # Get reference to a Logger
    logger = Logger(log_file_name=args['model_checkpoint_path']+'/logs/logger.logs', tb_log_dir=args['model_checkpoint_path']+'/logs/tb_logs')
    logger.log_dict_to_file(args)

    # Get Datasets to use
    dataset_names = None
    if args['train_dataset_file_names'] is not None:
        assert (args['validation_dataset_file_names'] is not None) and (args['test_dataset_file_names'] is not None), "both train, validation, and test dataset file names must be supplied if any one of them is specified for training!"
        dataset_names = { # Specify which files to use during training (useful for curriculum learning)
            'train': args['train_dataset_file_names'],
            'validation': args['validation_dataset_file_names'],
            'test': args['test_dataset_file_names']
        }

    # Load the saved tokenizer
    tokenizer = load_tokenizer_from_path(args['tokenizer_save_or_load_path'])
    logger.log_msg_to_console(f'Loaded tokenizer: {tokenizer}')

    # Load Model:
    model = CustomPreTrainingTransformerModel.load_saved_model(model_save_path=args['model_load_path'])
    model_size = sum(t.numel() for t in model.parameters())
    logger.log_to_file(f'Successfully loaded model: {model} \nwith size: {model_size/1000**2:.1f}M parameters')

    # Train the model
    if args['torch_training'] == 'True':  # Train the model using PyTorch Training Loop
        # Get Pretraining DataLoaders
        train_dataloader, validation_dataloader, test_dataloader = get_DataLoaders(dataset_names, tokenizer, task=args['pretraining_task'], batch_size=args['training_batch_size'], num_workers=args['num_workers'], return_small_debug_dataset=False)
        logger.log_msg_to_console(f'\nDataloaders:\n\ttrain_dataloader.length: {len(train_dataloader)},\n\tvalidation_dataloader.length: {len(validation_dataloader)},\n\ttest_dataloader.length: {len(test_dataloader)}')
        logger.log_to_file(f'\nDataloaders:\n\ttrain_dataloader.length: {len(train_dataloader)},\n\tvalidation_dataloader.length: {len(validation_dataloader)},\n\ttest_dataloader.length: {len(test_dataloader)}')

        # Train the model using pytorch training loop
        model = train_for_num_epochs_in_pytorch_loop(train_dataloader, model, args['num_epochs'], float(args['learning_rate']), args['grad_norm_clip'], validation_dataloader, args['model_checkpoint_path'], logger=logger)

    else: # Train the model using HuggingFace Trainer
        train_for_num_epochs_in_huggingface_trainer(dataset_names, model, tokenizer, args['pretraining_task'], args['training_batch_size'], args['num_epochs'], args['model_checkpoint_path']+"/hugginface_trainer", float(args['learning_rate']))

    logger.log_to_file('Training finished.')
    logger.log_msg_to_console('Training finished.')




# ================================================== 
# Load model, tokenizer and test :
if args['load_model_load_tokenizer_and_test'] == 'True':
    # Get reference to a Logger
    logger = Logger(log_file_name=args['model_checkpoint_path']+'/logs/logger.logs', tb_log_dir=args['model_checkpoint_path']+'/logs/tb_logs')
    logger.log_dict_to_file(args)

    # Get Datasets to use
    dataset_names = None
    if args['test_dataset_file_names'] is not None:
        dataset_names = { # Specify which files to use during training (useful for curriculum learning)
            'train': ['aochildes.train'],
            'validation': ['aochildes.dev'],
            'test': args['test_dataset_file_names']
        }

    # Load the saved tokenizer
    tokenizer = load_tokenizer_from_path(args['tokenizer_save_or_load_path'])
    logger.log_msg_to_console(f'Loaded tokenizer: {tokenizer}')

    # Load Model:
    model = CustomPreTrainingTransformerModel.load_saved_model(model_save_path=args['model_load_path'])
    model_size = sum(t.numel() for t in model.parameters())
    logger.log_to_file(f'Successfully loaded model: {model} \nwith size: {model_size/1000**2:.1f}M parameters')

    # Get Testing DataLoader
    _, _, test_dataloader = get_DataLoaders(dataset_names, tokenizer, task=args['pretraining_task'], batch_size=args['testing_batch_size'], num_workers=args['num_workers'], return_small_debug_dataset=False)
    logger.log_msg_to_console(f'\nDataloaders:\n\ttest_dataloader.length: {len(test_dataloader)}')
    logger.log_to_file(f'\nDataloaders:\n\ttest_dataloader.length: {len(test_dataloader)}')

    # Test the model:
    test_loss = test_model(test_dataloader, model, logger=logger)

    logger.log_to_file('Testing finished.')
    logger.log_msg_to_console('Testing finished.')