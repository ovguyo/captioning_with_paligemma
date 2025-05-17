
# Fine-tuning PaliGemma for Aerial Image Captioning

## Overview

This project includes scripts for CoT data generation, fine-tuning and evaluation of PaliGemma2.

## Dataset

* **RISC Dataset:** The RISC dataset (44,521 aerial images) was used to train and validate the fine-tuned PaliGemma model
* **Preprocessing:** Includes quote removal, spacing correction, capitalization standardization, punctuation standardization and filtering pool quality captions.
* **Splitting:** The dataset contains 'train', 'val' and 'test' splits.

## CoT Data Generation
Run `create_cot_steps.py` to create CoT sequences. The generated dataset can be dowloaded here:

https://drive.google.com/file/d/1Ppp3X-S5yDLQkWjDYlpx507vmD1nd7DW/view?usp=sharing


## Model & Training

* **Model:** Leverages pretrained PaliGemma 2 from Hugging Face Transformers via transfer learning.
* **Techniques:**
    * PyTorch framework.
    * AdamW optimizer.
    * Evaluation using NLP metrics such as BLEU, ROUGE, CIDEr.
    * Experiment tracking integrated with Weights & Biases (`wandb`).

## Status

* Further training and hyperparameter tuning are necessary to achieve optimal captioning performance.


