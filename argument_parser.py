import argparse
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
# train and save tokenizer
parser.add_argument("-ts_tok", "--train_and_save_tokenizer", default=False, help = "If True, a tokenizer will be trained and saved according to the arguments passed in.")
parser.add_argument("-l", "--length", type=int, help = "num rows from dataset to process while training the tokenizer.")
parser.add_argument("-vs", "--vocab_size", type=int, help = "newly trained tokenizer's target vocabulary size.")
parser.add_argument("-bs", "--batch_size", type=int, help = "batch_size parameter of tokenizer.train_new_from_iterator() function.")
parser.add_argument("-dn", "--dataset_names", type=dict, default=None, help = "names of the dataset files to load into dataset.")
parser.add_argument("-tok_pth", "--tokenizer_save_path", default="./save_dir/saved_tokenizer", help = "path to save the trained tokenizer.")
parser.add_argument("-tok_mdl", "--tokenizer_model", default="gpt2", help = "tokenizer model or path to load.")



# Add model arguments:
#
# Add training arguments:
#

# Parse the arguments
args = vars(parser.parse_args())

print(args)
print(args['image'])

# TODO: Train and save tokenizer
if args['train_and_save_tokenizer']:
    # train the tokenizer
    from pretraining_datasets import load_datasets_from_dir
    from tokenizer import train_tokenizer_on_corpus, save_tokenizer_to_path
    train_dataset = load_datasets_from_dir(dataset_names=None, streaming=False)['train']
    new_tokenizer = train_tokenizer_on_corpus(args['tokenizer_model'], args['length'], \
        args['vocab_size'], args['batch_size'], train_dataset)
    # save the trained tokenizer
    save_tokenizer_to_path(new_tokenizer, args['tokenizer_save_path'])
    print(f'Successfully trained and saved the tokenizer to path: {args["tokenizer_save_path"]}')


# TODO: Create Model from passed params via creating corresponding config file, and load tokenizer from a path.


# TODO: Load Model from path, and Load tokenizer form path.