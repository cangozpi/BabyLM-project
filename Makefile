# ------------ For loading a model's tokenizer, training it from scratch on a corpus, and saving it: 
train_tokenizer:
	python3 argument_parser.py --train_and_save_tokenizer True \
	--length 1000 --vocab_size 1000 --tokenizer_batch_size 20 --tokenizer_model_max_length 128 \
	--train_dataset_file_names aochildes.train bnc_spoken.train \
	--validation_dataset_file_names aochildes.dev \
	--test_dataset_file_names aochildes.test \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" --tokenizer_model gpt2
# --------------------------------------------------


# ------------ For Creating a model, loading a tokenizer, and training it on a dataset, then saving it: 
# creates a gpt2 model, pretrains it from scratch using causal language modeling (clm) objective:
create_gpt2_model_load_tokenizer_train_and_save_model_with_custom_torch_training_loop:
	python3 argument_parser.py --create_model_load_tokenizer_train_and_save_model True \
	--torch_training True \
	--train_dataset_file_names aochildes.train bnc_spoken.train \
	--validation_dataset_file_names aochildes.dev \
	--test_dataset_file_names aochildes.test \
	--transformer_model_name gpt2 --pretraining_task clm \
	--training_batch_size 16 --num_workers 0 \
	-lr "3e-5" --grad_norm_clip 1.0 --num_epochs 3 \
	--hidden_size 256 --num_attention_heads 4 --num_hidden_layers 4 \
	--model_checkpoint_path "./save_dir/training_loop_ckpt" 


# creates a bert-base-uncased model, pretrains it from scratch using masked language modeling (mlm) objective:
create_bert_model_load_tokenizer_train_and_save_model_with_custom_torch_training_loop:
	python3 argument_parser.py --create_model_load_tokenizer_train_and_save_model True \
	--torch_training True \
	--train_dataset_file_names aochildes.train bnc_spoken.train \
	--validation_dataset_file_names aochildes.dev \
	--test_dataset_file_names aochildes.test \
	--transformer_model_name bert-base-uncased --pretraining_task mlm \
	--training_batch_size 16 --num_workers 0 \
	-lr "3e-5" --grad_norm_clip 1.0 --num_epochs 3 \
	--hidden_size 256 --num_attention_heads 4 --num_hidden_layers 4 \
	--model_checkpoint_path "./save_dir/training_loop_ckpt" 
# --------------------------------------------------



# ------------ For Curriculum Learning (i.e. loading an already saved model and training it on a dataset again):
load_model_load_tokenizer_train_with_custom_torch_training_loop:
	python3 argument_parser.py --load_model_load_tokenizer_and_train True \
	--torch_training True \
	--train_dataset_file_names aochildes.train bnc_spoken.train \
	--validation_dataset_file_names aochildes.dev \
	--test_dataset_file_names aochildes.test \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" \
	--model_load_path "./save_dir/training_loop_ckpt/0" \
	--model_checkpoint_path "./save_dir/training_loop_resumed_ckpt" \
	--pretraining_task clm \
	--training_batch_size 16 --num_workers 0 \
	-lr "3e-5" --grad_norm_clip 1.0 --num_epochs 3 
# --------------------------------------------------


# ------------ For loading a model, loading a tokenizer, and testing it on a test_dataset: 
load_model_load_tokenizer_and_test:
	python3 argument_parser.py --load_model_load_tokenizer_and_test  True \
	--test_dataset_file_names aochildes.test \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" \
	--model_load_path "./save_dir/training_loop_ckpt/0" --pretraining_task clm \
	--model_checkpoint_path "./save_dir/testing" \
	--testing_batch_size 16 --num_workers 0
# --------------------------------------------------
	






# ------------ Extra info/examples: 
# An example for training a model using huggingface Trainer instead of my custom torch training loop implementation:
create_model_load_tokenizer_train_and_save_model_with_huggingface_trainer_loop:
	python3 argument_parser.py --create_model_load_tokenizer_train_and_save_model True \
	--torch_training False \
	--train_dataset_file_names aochildes.train bnc_spoken.train \
	--validation_dataset_file_names aochildes.dev \
	--test_dataset_file_names aochildes.test \
	--transformer_model_name gpt2 --pretraining_task clm \
	--training_batch_size 16 --num_workers 0 \
	-lr "5e-4" --grad_norm_clip 1.0 --num_epochs 3 \
	--hidden_size 256 --num_attention_heads 4 --num_hidden_layers 4 \
	--model_checkpoint_path "./save_dir/training_loop_ckpt" 