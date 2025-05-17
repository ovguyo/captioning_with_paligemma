import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict
import gc

from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from transformers import PaliGemmaForConditionalGeneration
from peft import PeftModel

# Import evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge_score import rouge_scorer
import nltk

try:
    nltk.download('punkt', quiet=True)
except:
    pass

# Optional: Install pycocoevalcap for CIDEr if you need it
# pip install pycocoevalcap
try:
    from pycocoevalcap.cider.cider import Cider

    CIDER_AVAILABLE = True
except ImportError:
    print("CIDEr metric not available. Install pycocoevalcap if needed.")
    CIDER_AVAILABLE = False

# Configuration
CHECKPOINT_PATH = "./paligemma-decoder-only-finetuned2/best_model"
BASE_MODEL_ID = "google/paligemma2-3b-mix-224"

# Dataset Configuration
CAPTION_ANNOTATIONS_FILE = '../dataset/RISCM/captions.csv'
IMAGE_FOLDER = '../dataset/RISCM/resized'
IMAGE_COL_NAME = 'image'
TARGET_CAPTION_COLUMNS = ['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']
SPLIT_COL_NAME = 'split'
TEST_SPLIT = 'test'  # or 'val' if you want to evaluate on validation set

# Evaluation Configuration
BATCH_SIZE = 1
MAX_GEN_LENGTH = 12
BASE_CAPTIONING_PROMPT_TEXT = "caption en"
OUTPUT_FILE = "evaluation_results.json"
DETAILED_OUTPUT_FILE = "detailed_evaluation_results.json"
LOAD_IN_4BIT = True
SEED = 42

# Generation Configuration
GENERATION_CONFIG = {
    "max_new_tokens": MAX_GEN_LENGTH,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 0.9,
    "repetition_penalty": 2.2,
    "length_penalty": 0.8,
    "no_repeat_ngram_size": 2,
}


def load_model_and_processor(checkpoint_path, base_model_id, load_in_4bit=True):
    """Load the fine-tuned model and processor."""
    print(f"Loading processor from {base_model_id}...")
    processor = AutoProcessor.from_pretrained(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Set pad token if not set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16

    # Quantization config
    bnb_config = None
    if load_in_4bit and device.type == 'cuda':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype,
        )

    print(f"Loading base model from {base_model_id}...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=model_dtype,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Loading LoRA adapters from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    # Ensure model's pad_token_id is aligned with tokenizer after loading
    if tokenizer.pad_token_id is not None and (
            not hasattr(model.config, 'pad_token_id') or model.config.pad_token_id is None):
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config,
               'eos_token_id') and model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    return model, processor, tokenizer, device, model_dtype


def load_test_data(annotations_file, image_folder, image_col, caption_cols, split_col, target_split):
    """Load test data from CSV."""
    print(f"Loading test data from {annotations_file}...")
    df = pd.read_csv(annotations_file)
    test_data = df[df[split_col] == target_split].reset_index(drop=True)
    print(f"Found {len(test_data)} examples for split '{target_split}'.")

    # Prepare data structure
    processed_data = []
    for idx, row in test_data.iterrows():
        image_filename = row[image_col]
        image_path = os.path.join(image_folder, image_filename)

        # Collect all available reference captions
        reference_captions = []
        for col in caption_cols:
            if pd.notna(row[col]) and str(row[col]).strip():
                reference_captions.append(str(row[col]).strip())

        if reference_captions:
            processed_data.append({
                'image_path': image_path,
                'image_filename': image_filename,
                'references': reference_captions
            })

    print(f"Processed {len(processed_data)} valid examples.")
    return processed_data


def generate_caption(model, processor, tokenizer, image_path, prompt, device, model_dtype, generation_config):
    """Generate a caption for a single image."""
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')

        # Process inputs
        inputs = processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=True
        )

        # Move to device
        pixel_values = inputs["pixel_values"].to(device)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        if device.type == 'cuda':
            pixel_values = pixel_values.to(model_dtype)

        # Generate caption
        with torch.no_grad():
            outputs = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )

        # Decode the generated caption
        # Remove the input tokens to get only the generated part
        generated_tokens = outputs[0][input_ids.shape[1]:]
        generated_caption = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(generated_caption)

        return generated_caption.strip()

    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return ""


def calculate_bleu_scores(predictions, references):
    """Calculate BLEU scores (1-4) for predictions."""
    # Individual BLEU scores
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []

    for pred, refs in zip(predictions, references):
        # Tokenize
        pred_tokens = pred.lower().split()
        refs_tokens = [ref.lower().split() for ref in refs]

        # Calculate individual BLEU scores
        bleu_1 = sentence_bleu(refs_tokens, pred_tokens, weights=(1, 0, 0, 0))
        bleu_2 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0))
        bleu_3 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.33, 0.33, 0.34, 0))
        bleu_4 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))

        bleu_1_scores.append(bleu_1)
        bleu_2_scores.append(bleu_2)
        bleu_3_scores.append(bleu_3)
        bleu_4_scores.append(bleu_4)

    # Calculate corpus BLEU as well
    pred_tokens_list = [pred.lower().split() for pred in predictions]
    refs_tokens_list = [[ref.lower().split() for ref in refs] for refs in references]

    corpus_bleu_4 = corpus_bleu(refs_tokens_list, pred_tokens_list)

    return {
        'bleu_1': np.mean(bleu_1_scores),
        'bleu_2': np.mean(bleu_2_scores),
        'bleu_3': np.mean(bleu_3_scores),
        'bleu_4': np.mean(bleu_4_scores),
        'corpus_bleu_4': corpus_bleu_4
    }


def calculate_rouge_scores(predictions, references):
    """Calculate ROUGE scores for predictions."""
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    for pred, refs in zip(predictions, references):
        # Get best score among all references
        best_rouge_1 = 0
        best_rouge_2 = 0
        best_rouge_l = 0

        for ref in refs:
            scores = rouge.score(pred, ref)
            best_rouge_1 = max(best_rouge_1, scores['rouge1'].fmeasure)
            best_rouge_2 = max(best_rouge_2, scores['rouge2'].fmeasure)
            best_rouge_l = max(best_rouge_l, scores['rougeL'].fmeasure)

        rouge_1_scores.append(best_rouge_1)
        rouge_2_scores.append(best_rouge_2)
        rouge_l_scores.append(best_rouge_l)

    return {
        'rouge_1': np.mean(rouge_1_scores),
        'rouge_2': np.mean(rouge_2_scores),
        'rouge_l': np.mean(rouge_l_scores)
    }


def calculate_cider_score(predictions, references):
    """Calculate CIDEr score if available."""
    if not CIDER_AVAILABLE:
        return {'cider': 'N/A (pycocoevalcap not installed)'}

    # Format for CIDEr
    gts = {}
    res = {}

    for i, (pred, refs) in enumerate(zip(predictions, references)):
        gts[i] = refs
        res[i] = [pred]

    cider = Cider()
    score, _ = cider.compute_score(gts, res)

    return {'cider': score}


def main():
    # Set seed for reproducibility
    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    # Load model and processor
    model, processor, tokenizer, device, model_dtype = load_model_and_processor(
        CHECKPOINT_PATH, BASE_MODEL_ID, LOAD_IN_4BIT
    )

    # Load test data
    test_data = load_test_data(
        CAPTION_ANNOTATIONS_FILE, IMAGE_FOLDER, IMAGE_COL_NAME,
        TARGET_CAPTION_COLUMNS, SPLIT_COL_NAME, TEST_SPLIT
    )

    # Generate captions for all test images
    predictions = []
    references = []
    detailed_results = []

    print(f"\nGenerating captions for {len(test_data)} images...")
    count = 0
    for item in tqdm(test_data, desc="Generating captions"):
        # Generate caption
        #if count > 15:
            #break
        generated_caption = generate_caption(
            model, processor, tokenizer,
            item['image_path'], BASE_CAPTIONING_PROMPT_TEXT,
            device, model_dtype, GENERATION_CONFIG
        )

        predictions.append(generated_caption)
        references.append(item['references'])

        # Store detailed results
        detailed_results.append({
            'image': item['image_filename'],
            'generated': generated_caption,
            'references': item['references']
        })

        # Clear GPU cache periodically
        if len(predictions) % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        count = count + 1

    # Calculate metrics
    print("\nCalculating metrics...")
    bleu_scores = calculate_bleu_scores(predictions, references)
    rouge_scores = calculate_rouge_scores(predictions, references)
    cider_score = calculate_cider_score(predictions, references)

    # Combine all metrics
    all_metrics = {
        **bleu_scores,
        **rouge_scores,
        **cider_score,
        'num_samples': len(predictions)
    }

    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Saving detailed results to {DETAILED_OUTPUT_FILE}...")
    with open(DETAILED_OUTPUT_FILE, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    # Print summary
    print("\n=== Evaluation Results ===")
    for metric, value in all_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # Show a few examples
    print("\n=== Sample Predictions ===")
    for i in range(min(5, len(detailed_results))):
        result = detailed_results[i]
        print(f"\nImage: {result['image']}")
        print(f"Generated: {result['generated']}")
        print(f"Reference 1: {result['references'][0]}")
        if len(result['references']) > 1:
            print(f"Reference 2: {result['references'][1]}")
        print("-" * 50)


if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nEvaluation completed!")