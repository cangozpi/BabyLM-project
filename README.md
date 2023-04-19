# BabyLM Challenge Project: 

## Summary: 
This shared task challenges community members to train a language model from
scratch on the same amount of linguistic data available to a child. Submissions should be
implemented in Huggingface's Transformers library and will be evaluated on a shared
pipeline. For the details of the challange, refer to https://babylm.github.io/ .

---

### TODO:

1. Dataset&Dataloaders 🔨
2. Load provided baseline models
3. Run provided evaluation script from command line
4. Evaluate loaded baseline models using the provided evaluation pipelines from the command line
5. Train tokenizer on the training data
6. Implement a custom model 
7. Try to train your custom model using your tokenizer and evaluate it
8. Find papers, create roadmap
9. Write proposal

---

### Key Notes:
* Submissions are trained exclusively on the provided dataset.
* The pipeline assumes all models can be loaded
and queried in HuggingFace’s transformers library. Additionally, all models must be able to score a sequence—e.g., assign a log-likelihood or pseudo log-likelihood — and must be able to be fine-tuned to perform classification tasks. Models do not need to be able to generate sequences. Submissions must include model outputs for each of the core evaluations in a format that we specify in our evaluation pipeline.

---