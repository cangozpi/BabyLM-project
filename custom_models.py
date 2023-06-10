# from transformers import DistilBertForQuestionAnswering
from transformers import GPT2PreTrainedModel, BertModel, GPT2Model, GPT2ForSequenceClassification, AutoConfig, PreTrainedModel
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
from transformers import AutoModel, PretrainedConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
from os.path import join

class CustomPreTrainingTransformerConfig(PretrainedConfig):
    model_type = "CustomPreTrainingTransformer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CustomPreTrainingTransformerModel(PreTrainedModel):
    config_class = CustomPreTrainingTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        assert pretraining_task in ['clm', 'mlm']
        self.num_labels = config.num_labels

        self.encoder = AutoModel.from_config(config)

        if config.pretraining_task == 'clm':
            self.head = AutoModelForCausalLM.from_config(config).lm_head
        elif config.pretraining_task == 'mlm':
            self.head = AutoModelForMaskedLM.from_config(config).cls

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Use model body to get encoder representations
        outputs = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        last_hidden_state = outputs['last_hidden_state']
        # Apply classification head to encoder representations
        logits = self.head(last_hidden_state)
        # Calculate loss
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.reshape(-1, self.num_labels), labels.reshape(-1))
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
        attentions=outputs.attentions)
    

    def save_model_and_config(self, model_save_path):
        """
        Save model config and model state dict to the specified path
        """
        # Save model config
        self.config.save_pretrained(model_save_path)
        # Save model weights
        torch.save(self.state_dict(), join(model_save_path, "model_state_dict"))
    

    @staticmethod
    def load_saved_model(model_save_path):
        """
        Load model config and model state dict from the specified path
        """
        # Load model config
        config = AutoConfig.from_pretrained(model_save_path)
        # Initialize a model from the config
        model = CustomPreTrainingTransformerModel(config)
        # Load model weights
        model.load_state_dict(torch.load(join(model_save_path, "model_state_dict")))

        return model


if __name__ == "__main__":
    # from transformers import AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    task_index = 1

    from transformers import AutoTokenizer
    model_name = ['gpt2', 'bert-base-uncased'][task_index]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Choose pretraining task
    pretraining_task = ['clm', 'mlm'][task_index]

    # Create Config for the transformer model
    # config = AutoConfig.from_pretrained("bert-base-uncased")
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = tokenizer.vocab_size # make model's vocabulary size match the tokenizer's vocab size
    config.pretraining_task = pretraining_task
    # config['add_pooling_layer'] = False

    # Initialize Model from the config for the specified pretraining_task
    model = CustomPreTrainingTransformerModel(config)

    print(config)
    print(model.head)


    # -------------------- Test custom model's forward pass:

    train_dataset_names = { # Specify which files to use during training (useful for curriculum learning)
        'train': ['aochildes.train'],
        'validation': ['aochildes.dev'],
        'test': ['aochildes.test']
    }
    from pretraining_datasets import get_DataLoaders
    from pytorch_training_loop import train_for_num_epochs_in_pytorch_loop
    train_dataloader, _, _ = get_DataLoaders(train_dataset_names, tokenizer, task='clm', batch_size=2, num_workers=0)

    # Train the model using pytorch training loop
    # model = train_for_num_epochs_in_pytorch_loop(train_dataloader, model, 1)


    # -------------------- Test saving and loading custom model:
    # Save custom model
    model_save_path = './save_dir/saved_CustomPreTrainingTransformerModel'
    # do your finetuning and save your custom model
    # model.save_pretrained(model_save_path)
    model.save_model_and_config(model_save_path)

    # Load custom model
    # del model
    # CustomPreTrainingTransformerModel.register_for_auto_class()
    # CustomPreTrainingTransformerConfig.register_for_auto_class()
    # Register your config and your model
    # AutoConfig.register("CustomPreTrainingTransformer", CustomPreTrainingTransformerConfig)
    # AutoModel.register(CustomPreTrainingTransformerConfig, CustomPreTrainingTransformerModel)
    loaded_model = CustomPreTrainingTransformerModel.load_saved_model(model_save_path)
    print(f'Loaded model: {loaded_model}')

    # Test that loaded_model and model is the same (weights, biases, ...)
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        flag = torch.allclose(p1, p2)
        if flag == False:
            raise Exception('loaded_model is different from the saved_model!')
    print('Loaded model and the saved model are the same.')

    # loaded_model = AutoModel.from_pretrained(model_save_path)