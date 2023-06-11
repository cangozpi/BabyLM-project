train_tokenizer:
	python3 argument_parser.py --train_and_save_tokenizer True \
	--length 1000 --vocab_size 1000 --tokenizer_batch_size 20 \
	--train_dataset_file_names aochildes.train bnc_spoken.train \
	--validation_dataset_file_names aochildes.dev \
	--test_dataset_file_names aochildes.test \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" --tokenizer_model gpt2

create_model_load_tokenizer_train_and_save_model_with_custom_torch_training_loop:
	python3 argument_parser.py --create_model_load_tokenizer_train_and_save_model True \
	--torch_training True \
	--train_dataset_file_names aochildes.train bnc_spoken.train \
	--validation_dataset_file_names aochildes.dev \
	--test_dataset_file_names aochildes.test \
	--transformer_model_name gpt2 --pretraining_task clm \
	--training_batch_size 16 --num_workers 0 \
	-lr "3e-5" --grad_norm_clip 1.0 --num_epochs 3 \
	--model_checkpoint_path "./save_dir/training_loop_ckpt" 

create_model_load_tokenizer_train_and_save_model_with_huggingface_trainer_loop:
	python3 argument_parser.py --create_model_load_tokenizer_train_and_save_model True \
	--torch_training False \
	--train_dataset_file_names aochildes.train bnc_spoken.train \
	--validation_dataset_file_names aochildes.dev \
	--test_dataset_file_names aochildes.test \
	--transformer_model_name gpt2 --pretraining_task clm \
	--training_batch_size 16 --num_workers 0 \
	-lr "5e-4" --grad_norm_clip 1.0 --num_epochs 3 \
	--model_checkpoint_path "./save_dir/training_loop_ckpt" 

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
	-lr "3e-5" --grad_norm_clip 1.0 --num_epochs 3 \
	