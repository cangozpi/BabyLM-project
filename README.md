# BabyLM Challenge Project: 

## Summary: 
This shared task challenges community members to train a language model from
scratch on the same amount of linguistic data available to a child. Submissions should be
implemented in Huggingface's Transformers library and will be evaluated on a shared
pipeline. For the details of the challange, refer to https://babylm.github.io/ .

---

### TODO:

1. Dataset&Dataloaders ✅
2. Load provided baseline models and tokenizers ✅
3. Training with Huggingface PyTorch Trainer ✅
4. Training in Native Pytorch 🔨
5. Overfit provided baseline model on the training dataset 
6. Run provided evaluation script from command line 
7. Evaluate loaded baseline models using the provided evaluation pipelines from the command line
8. Train tokenizer on the training data from scratch
9. Implement a custom model 
10. Try to train your custom model using your tokenizer and evaluate it
11. Find papers and create a roadmap
12. Write proposal

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
    ```bash
    python3 main.py
    ```

---

### Key Notes:
* Submissions are trained exclusively on the provided dataset.
* The pipeline assumes all models can be loaded
and queried in HuggingFace’s transformers library. Additionally, all models must be able to score a sequence—e.g., assign a log-likelihood or pseudo log-likelihood — and must be able to be fine-tuned to perform classification tasks. Models do not need to be able to generate sequences. Submissions must include model outputs for each of the core evaluations in a format that we specify in our evaluation pipeline.

---