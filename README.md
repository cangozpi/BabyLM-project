# BabyLM Challenge Project: 

## Summary: 
This shared task challenges community members to train a language model from
scratch on the same amount of linguistic data available to a child. Submissions should be
implemented in Huggingface's Transformers library and will be evaluated on a shared
pipeline. For the details of the challange, refer to https://babylm.github.io/ .

---
### Run code:
* Install dataset
    ```bash
    wget https://github.com/babylm/babylm.github.io/raw/main/babylm_data.zip
    unzip babylm_data.zip -d ./babylm_data
    ```
* Install python dependencies
    ```bash
        pip install -r requirements.txt
    ```
* Running code:

    To reproduce our experiments call the command with the corresponding _step_ name in the Makefile. Check out Makefile and run.py for details.
    ```bash
    make step<no>
    ```
    For example to train a small tokenizer first run:
    ```bash
    make step1
    ```

---

### Implementation Details:
File Structure:
* _argument\_parser.py_: Adds command line argument parsing capability to our implementation. Using the arguments defined in this file, we can perform our experiments easily. It allows one to specify tokenizers to use with models, or train tokenizers from scratch. Save and load tokenizers and models from both HuggingFace and from local directories. Configure most of the hyperparameters of the models, and specify which datasets to use for training, validating, and testing the models. It is combined with \emph{Makefile} to ease the experimentation process and contribute to the reproduciblity of our work. An example for its use can be found in \ref{fig:makefile_example}.

* _custom\_models.py_: It implements the CustomPretrainingTransformerModel class which we used to create, save, and load our custom transformer models. It extends PreTrainedModel class of HuggingFace which extends torch.nn.Module class underneath. It can load transformer models from HuggingFace and use their body as its own encoder. It also initializes a linear layer which is to be used as the classification head over the target vocabulary.

* huggingface\_trainer\_loop.py_: It adds support for training the models using the HuggingFace's Trainer. Our implementation supports both training using huggingface Trainer, and using our custom training loop written in PyTorch. These two options can be used interchangebly by simply passing the --torch\_training parameter supported by the argument\_parser.py. In all of our experiments we have used our custom PyTorch training loop.  

* _Makefile_: Makefile is not an integral part of our code implementation. In other words removal of it would not cause any harm. We have used it for the sole purpose of having an easier way of configuring the command line arguments that are supported by argument\_parser.py. We have taken it a step further by extending with all of the python calls one needs to reproduce our experiments. It has commands that correspond to the names of the steps of our experiments (e.g. Step 1, Step 7\_1).

* _pretraining\_dataset.py_: It loads specified datasets and creates training, validation, and testing datasets. Given a tokenizer and a dataset, it tokenizes the datasets. In addition to these it is responsible for the implementation of the data collators for the causal language modeling objective and the masked language modeling objective. It most notably relies on the HuggingFace's datasets module and its supported map functionality.

* _pytorch\_training\_loop.py: It is a replacement for the huggingface\_trainer\_loop.py functionality. We found HuggingFace's Trainer implementation to be hiding too much from us and hard to extend, thus we decided to implement a training loop ourselves using PyTorch. It is used for training a model, evaluating it after every epoch. It keeps track of the model with the best validation performance and saves it as checkpoint. This acts in a similar manner to early stopping. It defines an torch.optim.AdamW optimizer, and implements gradient normalization to train the model with the hopes of avoiding exploding gradients problem, and calls an instance of our custom Logger class to log the appropriate information on the supported mediums. It also implements the functionality for testing a model on a given test dataset.

* _tokenizer.py_: It adds the functionality to load, save, train tokenizers from both HuggingFace and local paths.

* _utils.py_: Motivated by the fact that training transformer models have high time and compute requirements which we were struggling with, we decided that logging was of uppermost importance to us. To this end, we have developed this module which implements a custom Logger class. This class extends python's logging module and makes use of TensorBoard. It supports logging to command line, files, and TensorBoard logs. We made extensive use of it in order to have a better understanding of the learning dynamics that are taking place in our experiments, to finding bugs in our code during the earlier stages of its development phase, and for keeping track of the hyperparameters we have used during our experiments. Refer to. In our experiments it creates a directory called save\_dir, and logs the experiments under their corresponding name. For instance, running Step2\_1 would create the directory ./save\_dir/step\_1. It is this directory in which we have also saved our model checkpoints during the training.
    

---
### Key Notes:
* Submissions are trained exclusively on the provided dataset.
* The pipeline assumes all models can be loaded
and queried in HuggingFace’s transformers library. Additionally, all models must be able to score a sequence—e.g., assign a log-likelihood or pseudo log-likelihood — and must be able to be fine-tuned to perform classification tasks. Models do not need to be able to generate sequences. Submissions must include model outputs for each of the core evaluations in a format that we specify in our evaluation pipeline.

---