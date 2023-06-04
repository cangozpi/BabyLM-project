# from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import T5Tokenizer, T5Model, RobertaModel, RobertaTokenizer, RobertaConfig
from torch.utils.data import DataLoader
from dataset_StrictSmall import load_datasets_from_dir
from dataset_StrictSmall import load_datasets_from_dir, load_dataset, pre_process_dataset
from dataset_StrictSmall import load_datasets_from_dir, pre_process_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tokenizers.trainers import BpeTrainer
from transformers import RobertaForMaskedLM
from transformers import AdamW
from transformers import get_scheduler
from native_pytorch_trainer import train_in_native_pytorch


def tokenize():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    train_data, dataset = load_datasets_from_dir()
    trainer = BpeTrainer(vocab_size=30_522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(train_data, trainer)
    tokenizer.model.save("data/")
    tokenizer.save("data/tokenizer.json")


#tokenize()

# tokenizer = T5Tokenizer.from_pretrained("data/tokenizer-dataset2.json")
# from transformers import T5Config
# config = T5Config(vocab_size=32128,
#                   d_model=512,
#                   d_kv=64,
#                   d_ff=2048,
#                   num_layers=6,
#                   num_decoder_layers=None,
#                   num_heads=8,
#                   relative_attention_num_buckets=32,
#                   relative_attention_max_distance=128,
#                   dropout_rate=0.1,
#                   layer_norm_epsilon=1e-6,
#                   initializer_factor=1.0,
#                   feed_forward_proj="relu",
#                   is_encoder_decoder=True,
#                   use_cache=True)
# model = T5Model(config)


tokenizer = RobertaTokenizer.from_pretrained("data/")
config = RobertaConfig(
    vocab_size=30_522,
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

model = RobertaForMaskedLM(config)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print(device)

# TODO tokenize train dataset and create dataloaders


batch_size = 16  # Batch size used for DataLoader
max_seq_length = 20  # fixed length of the sequences (i.e. num tokens per entry)
map_batch_size = 1000  # batch size used for dataset.map() function during pre-processing
num_proc = 4

train_data, dataset = load_datasets_from_dir()
tokenized_dataset = pre_process_dataset(dataset, tokenizer, max_seq_length, map_batch_size, num_proc)

# setting up Loaders

train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=16, shuffle=True)
test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=16, shuffle=False)

# setting up model
#model = AutoModelForSeq2SeqLM.from_pretrained("babylm/t5-base-strict")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# setting up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.to(device)
print(device)

# train
model.train()
num_epochs = 1
# getting the bacthes in the train dataloader and calculating loss
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        # model parameters given a transformer model
        outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
print("epoch: {}, loss: {}".format(epoch, loss.item()))
