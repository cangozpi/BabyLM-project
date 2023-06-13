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
	






# ------------ Some Utilities: 
# ------------ For downloading and extracting the dataset files:
download_and_extract_dataset:
	wget https://github.com/babylm/babylm.github.io/raw/main/babylm_data.zip
	unzip babylm_data.zip -d babylm_data


# ------------ For starting TensorBoard:
start_tensorboard:
	tensorboard --logdir save_dir/








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





# ====================================================================== FOR EXPERIMENTS:
all_training_dataset_file_names=aochildes.train gutenberg.train switchboard.train bnc_spoken.train open_subtitles.train wikipedia.train cbt.train qed.train children_stories.train  simple_wikipedia.train
all_dev_dataset_file_names=aochildes.dev gutenberg.dev switchboard.dev bnc_spoken.dev open_subtitles.dev wikipedia.dev cbt.dev qed.dev children_stories.dev  simple_wikipedia.dev
all_test_dataset_file_names=aochildes.test gutenberg.test switchboard.test bnc_spoken.test open_subtitles.test wikipedia.test cbt.test qed.test children_stories.test  simple_wikipedia.test
easier_training_dataset_file_names=aochildes.train open_subtitles.train qed.train simple_wikipedia.train switchboard.train
harder_training_dataset_file_names=bnc_spoken.train cbt.train children_stories.train gutenberg.train wikipedia.train

step1: # Train a small tokenizer
	python3 argument_parser.py --train_and_save_tokenizer True \
	--length 10000 --vocab_size 2000 --tokenizer_batch_size 20 --tokenizer_model_max_length 128 \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" --tokenizer_model gpt2

step2_1: # Train the small GPT model
	python3 argument_parser.py --create_model_load_tokenizer_train_and_save_model True \
	--torch_training True \
	--transformer_model_name gpt2 --pretraining_task clm \
	--training_batch_size 16 --num_workers 0 \
	-lr "3e-5" --grad_norm_clip 1.0 --num_epochs 3 \
	--hidden_size 256 --num_attention_heads 4 --num_hidden_layers 4 \
	--model_checkpoint_path "./save_dir/step2_1/training_loop_ckpt/small_gpt_whole_datasets" 

step2_2:# Test the trained small GPT model 
	python3 argument_parser.py --load_model_load_tokenizer_and_test  True \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" \
	--model_load_path "./save_dir/step2_1/training_loop_ckpt/small_gpt_whole_datasets/2" --pretraining_task clm \
	--model_checkpoint_path "./save_dir/step2_2/testing" \
	--testing_batch_size 16 --num_workers 0

step3_1: # Train using curriculum learning approach 1(train on easier data first)
	python3 argument_parser.py --create_model_load_tokenizer_train_and_save_model True \
	--torch_training True \
	--train_dataset_file_names ${easier_training_dataset_file_names} \
	--validation_dataset_file_names ${all_dev_dataset_file_names} \
	--test_dataset_file_names ${all_test_dataset_file_names} \
	--transformer_model_name gpt2 --pretraining_task clm \
	--training_batch_size 16 --num_workers 0 \
	-lr "3e-5" --grad_norm_clip 1.0 --num_epochs 3 \
	--hidden_size 256 --num_attention_heads 4 --num_hidden_layers 4 \
	--model_checkpoint_path "./save_dir/step3_1/training_loop_ckpt/small_gpt_easy_datasets" 

step3_2: # Train using curriculum learning approach 1 cont.(train on harder data now)
	python3 argument_parser.py --load_model_load_tokenizer_and_train True \
	--torch_training True \
	--train_dataset_file_names ${harder_training_dataset_file_names} \
	--validation_dataset_file_names ${all_dev_dataset_file_names} \
	--test_dataset_file_names ${all_test_dataset_file_names} \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" \
	--model_load_path "./save_dir/step3_1/training_loop_ckpt/small_gpt_easy_datasets/2" \
	--model_checkpoint_path "./save_dir/step3_2/training_loop_resumed_ckpt/small_gpt_hard_datasets" \
	--pretraining_task clm \
	--training_batch_size 16 --num_workers 0 \
	-lr "3e-5" --grad_norm_clip 1.0 --num_epochs 3 

step3_3: # Test the small GPT model with curriculum learning
	python3 argument_parser.py --load_model_load_tokenizer_and_test  True \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" \
	--model_load_path "./save_dir/step3_2/training_loop_resumed_ckpt/small_gpt_hard_datasets/2" --pretraining_task clm \
	--model_checkpoint_path "./save_dir/step3_3/testing" \
	--testing_batch_size 16 --num_workers 0

step4_1: # Train a small BERT model
	python3 argument_parser.py --create_model_load_tokenizer_train_and_save_model True \
	--torch_training True \
	--transformer_model_name bert-base-uncased --pretraining_task mlm \
	--training_batch_size 16 --num_workers 0 \
	-lr "3e-5" --grad_norm_clip 1.0 --num_epochs 3 \
	--hidden_size 256 --num_attention_heads 4 --num_hidden_layers 4 \
	--model_checkpoint_path "./save_dir/step4_1/training_loop_ckpt/small_bert_whole_datasets" 


step4_2: # Test the small BERT model
	python3 argument_parser.py --load_model_load_tokenizer_and_test  True \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" \
	--model_load_path "./save_dir/step4_1/training_loop_ckpt/small_bert_whole_datasets/2" --pretraining_task mlm \
	--model_checkpoint_path "./save_dir/step4_2/testing" \
	--testing_batch_size 16 --num_workers 0

step5_1: # Train using curriculum learning approach 1(train on easier data first)
	python3 argument_parser.py --create_model_load_tokenizer_train_and_save_model True \
	--torch_training True \
	--train_dataset_file_names ${easier_training_dataset_file_names} \
	--validation_dataset_file_names ${all_dev_dataset_file_names} \
	--test_dataset_file_names ${all_test_dataset_file_names} \
	--transformer_model_name bert-base-uncased --pretraining_task mlm \
	--training_batch_size 16 --num_workers 0 \
	-lr "3e-5" --grad_norm_clip 1.0 --num_epochs 3 \
	--hidden_size 256 --num_attention_heads 4 --num_hidden_layers 4 \
	--model_checkpoint_path "./save_dir/step5_1/training_loop_ckpt/small_bert_easy_datasets" 

step5_2: # Train using curriculum learning approach 1 cont.(train on harder data now)
	python3 argument_parser.py --load_model_load_tokenizer_and_train True \
	--torch_training True \
	--train_dataset_file_names ${harder_training_dataset_file_names} \
	--validation_dataset_file_names ${all_dev_dataset_file_names} \
	--test_dataset_file_names ${all_test_dataset_file_names} \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" \
	--model_load_path "./save_dir/step5_1/training_loop_ckpt/small_bert_easy_datasets/2" \
	--model_checkpoint_path "./save_dir/step5_2/training_loop_resumed_ckpt/small_bert_hard_datasets" \
	--pretraining_task mlm \
	--training_batch_size 16 --num_workers 0 \
	-lr "3e-5" --grad_norm_clip 1.0 --num_epochs 3 

step5_3: # Test the small BERT model with curriculum learning
	python3 argument_parser.py --load_model_load_tokenizer_and_test  True \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" \
	--model_load_path "./save_dir/step5_2/training_loop_resumed_ckpt/small_bert_hard_datasets/2" --pretraining_task mlm \
	--model_checkpoint_path "./save_dir/step5_3/testing" \
	--testing_batch_size 16 --num_workers 0



best_model_name=gpt2 # choose the best model model to scale up
best_model_pretraining_task=clm


step6_1: # Scale up best model (e.g. increase hidden_dims, batch_size, learning rate, and train longer)
	python3 argument_parser.py --create_model_load_tokenizer_train_and_save_model True \
	--torch_training True \
	--transformer_model_name ${best_model_name} --pretraining_task ${best_model_pretraining_task} \
	--training_batch_size 32 --num_workers 0 \
	-lr "1e-4" --grad_norm_clip 1.0 --num_epochs 6 \
	--hidden_size 256 --num_attention_heads 8 --num_hidden_layers 5 \
	--model_checkpoint_path "./save_dir/step6_1/training_loop_ckpt/scaled_model_whole_datasets" 

step6_2: # Test the scaled up best model
	python3 argument_parser.py --load_model_load_tokenizer_and_test  True \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" \
	--model_load_path "./save_dir/step6_1/training_loop_resumed_ckpt/scaled_model_whole_datasets/5" \
	--pretraining_task ${best_model_pretraining_task} \
	--model_checkpoint_path "./save_dir/step6_2/testing" \
	--testing_batch_size 16 --num_workers 0


step7_1: # Scale up best model and train using curriculum learning approach 1(train on easier data first)
	python3 argument_parser.py --create_model_load_tokenizer_train_and_save_model True \
	--torch_training True \
	--train_dataset_file_names ${easier_training_dataset_file_names} \
	--validation_dataset_file_names ${all_dev_dataset_file_names} \
	--test_dataset_file_names ${all_test_dataset_file_names} \
	--transformer_model_name ${best_model_name} --pretraining_task ${best_model_pretraining_task} \
	--training_batch_size 32 --num_workers 0 \
	-lr "1e-4" --grad_norm_clip 1.0 --num_epochs 6 \
	--hidden_size 256 --num_attention_heads 8 --num_hidden_layers 5 \
	--model_checkpoint_path "./save_dir/step7_1/training_loop_ckpt/scaled_model_easy_datasets" 

step7_2: # Scaled up moodel: Train using curriculum learning approach 1 cont.(train on harder data now)
	python3 argument_parser.py --load_model_load_tokenizer_and_train True \
	--torch_training True \
	--train_dataset_file_names ${harder_training_dataset_file_names} \
	--validation_dataset_file_names ${all_dev_dataset_file_names} \
	--test_dataset_file_names ${all_test_dataset_file_names} \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" \
	--model_load_path "./save_dir/step7_1/training_loop_ckpt/scaled_model_easy_datasets/5" \
	--model_checkpoint_path "./save_dir/step7_2/training_loop_resumed_ckpt/scaled_model_hard_datasets" \
	--pretraining_task ${best_model_pretraining_task} \
	--training_batch_size 32 --num_workers 0 \
	-lr "1e-4" --grad_norm_clip 1.0 --num_epochs 6 

step7_3:  # Test the scaled up best model trained using curriculum learning
	python3 argument_parser.py --load_model_load_tokenizer_and_test  True \
	--tokenizer_save_or_load_path "./save_dir/saved_tokenizer" \
	--model_load_path "./save_dir/step7_2/training_loop_resumed_ckpt/scaled_model_hard_datasets/5" \
	--pretraining_task ${best_model_pretraining_task} \
	--model_checkpoint_path "./save_dir/step7_3/testing" \
	--testing_batch_size 16 --num_workers 0