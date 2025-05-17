import os
import random
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import gc

from transformers import PaliGemmaForConditionalGeneration
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoProcessor

from peft import LoraConfig, get_peft_model, TaskType

import wandb

# Placeholder Hyperparameters
LEARNING_RATE = 1e-4  # Lower learning rate for stability
NUM_EPOCHS = 3
BATCH_SIZE = 1
OUTPUT_DIR = "./paligemma-decoder-only-finetuned3"  # New output directory
LOG_INTERVAL = 10

# Model and Dataset Configuration
MODEL_ID = "google/paligemma2-3b-mix-224"
CAPTION_ANNOTATIONS_FILE = '../dataset/RISCM/processed_captions.csv'
IMAGE_FOLDER = '../dataset/RISCM/resized'
IMAGE_COL_NAME = 'image'
# Define columns for target captions
TARGET_CAPTION_COLUMNS = ['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']
SPLIT_COL_NAME = 'split'
TRAIN_TARGET_SPLIT = 'train'
VAL_TARGET_SPLIT = 'val'

# AutoProcessor will prepend <image> tokens and <bos>
BASE_CAPTIONING_PROMPT_TEXT = "caption en"

# Limit Training Samples
MAX_TRAIN_SAMPLES = 30000
MAX_VAL_SAMPLES = 2000
MAX_TARGET_CAPTION_LENGTH = 32
IGNORE_INDEX = -100
SEED = 42

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.15
LORA_TARGET_MODULES = [
    # Specific targeting of final decoder layers
    "model.decoder.layers.11.self_attn.q_proj",
    "model.decoder.layers.11.self_attn.k_proj",
    "model.decoder.layers.11.self_attn.v_proj",
    "model.decoder.layers.11.self_attn.o_proj",
    "model.decoder.layers.10.self_attn.q_proj",
    "model.decoder.layers.10.self_attn.k_proj",
    "model.decoder.layers.10.self_attn.v_proj",
    "model.decoder.layers.10.self_attn.o_proj",
    "lm_head"
]

LOAD_IN_4BIT = True


# --- Dataset Class for Image Captioning ---
class ImageCaptioningDataset(Dataset):
    def __init__(self, annotations_file, image_folder, image_col, target_caption_cols, split_col, target_split,
                 max_samples=None):
        self.image_folder = image_folder
        self.image_col = image_col
        self.target_caption_cols = target_caption_cols

        print(f"Dataset ({target_split}): Loading caption annotations from {annotations_file}...")
        df = pd.read_csv(annotations_file)
        print(f"Dataset ({target_split}): Filtering for '{target_split}' split...")
        self.data = df[df[split_col] == target_split].reset_index(drop=True)
        print(f"Dataset ({target_split}): Found {len(self.data)} initial examples for split '{target_split}'.")

        if max_samples is not None and max_samples > 0:
            if max_samples < len(self.data):
                print(f"Dataset ({target_split}): Limiting data to first {max_samples} samples.")
                self.data = self.data.head(max_samples)
            else:
                print(
                    f"Dataset ({target_split}): Requested max_samples ({max_samples}) is >= available samples ({len(self.data)}). Using all available.")

        if len(self.data) == 0:
            raise ValueError(
                f"Dataset ({target_split}): No data found or remaining after limiting in {annotations_file} for split '{target_split}'.")
        print(f"Dataset ({target_split}): Using {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of bounds for limited dataset")
        row = self.data.iloc[idx]
        image_filename = row[self.image_col]
        image_path = os.path.join(self.image_folder, image_filename)

        # Randomly select one of the target captions
        available_captions = [str(row[col]) for col in self.target_caption_cols if
                              pd.notna(row[col]) and str(row[col]).strip()]
        if not available_captions:
            return None  # Skip if no valid captions for this row
        target_caption = random.choice(available_captions)

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            return None
        except Exception:
            return None

        # Return image, the text prompt, and the selected target caption
        return image, BASE_CAPTIONING_PROMPT_TEXT, target_caption


# --- Collate Function for Image Captioning ---
def collate_fn_captioning(batch, processor, tokenizer):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    image, text_prompt, target_caption = batch[0]

    try:
        # The processor will add image tokens and format it correctly
        model_inputs = processor(
            text=[text_prompt],
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=True
        )
    except Exception as e:
        print(f"Collate Error: Processor failed: {e}")
        return None

    pixel_values = model_inputs["pixel_values"]
    prefix_input_ids = model_inputs["input_ids"]
    prefix_attn_mask = model_inputs["attention_mask"]

    try:
        # Tokenize the target caption
        target_tokens = tokenizer(
            target_caption,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=MAX_TARGET_CAPTION_LENGTH
        )
        target_input_ids = target_tokens["input_ids"]
        target_attn_mask = target_tokens["attention_mask"]
    except Exception as e:
        print(f"Collate Error: Tokenizer failed for target caption '{target_caption[:50]}...': {e}")
        return None

    # For captioning, concatenate prefix and target for teacher forcing
    input_ids = torch.cat([prefix_input_ids, target_input_ids], dim=1)
    attention_mask = torch.cat([prefix_attn_mask, target_attn_mask], dim=1)

    # Labels: Mask the prefix (image tokens + captioning_prompt tokens)
    prefix_len = prefix_input_ids.shape[1]
    labels = input_ids.clone()
    labels[:, :prefix_len] = IGNORE_INDEX
    labels[labels == tokenizer.pad_token_id] = IGNORE_INDEX

    final_batch = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

    return final_batch


# --- Main Training Function ---
def main():
    if SEED is not None:
        random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    wandb.init(
        project="paligemma-decoder-only",
        config={
            "learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
            "model_id": MODEL_ID, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT, "lora_target_modules": LORA_TARGET_MODULES,
            "max_train_samples": MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES is not None else "all",
            "max_val_samples": MAX_VAL_SAMPLES if MAX_VAL_SAMPLES is not None else "all",
            "max_target_caption_length": MAX_TARGET_CAPTION_LENGTH,
            "base_captioning_prompt": BASE_CAPTIONING_PROMPT_TEXT,
            "load_in_4bit": LOAD_IN_4BIT, "seed": SEED,
            "training_approach": "decoder_only"
        }
    )
    wandb.define_metric("train/step_loss", step_metric="trainer/global_step")
    wandb.define_metric("train/epoch_avg_loss", step_metric="epoch")
    wandb.define_metric("val/epoch_avg_loss", step_metric="epoch")

    print("--- Starting Decoder-Only Fine-tuning Script with LoRA & Validation ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    bnb_config = None
    if LOAD_IN_4BIT and device.type == 'cuda':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_use_double_quant=True
        )
        print("BitsAndBytesConfig for 4-bit loading prepared.")

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=model_dtype,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print("Base model loaded successfully.")

    # Explicitly freeze vision tower and multi-modal projector
    # This is crucial to avoid repetitive tokens
    print("Freezing vision tower and multi-modal projector...")
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.multi_modal_projector.parameters():
        param.requires_grad = False


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side='left', model_max_length=1024)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model, 'config') and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id
        print(f"Tokenizer pad_token set to eos_token: {tokenizer.pad_token}")
    print("Tokenizer loaded.")

    print("Configuring LoRA for decoder-only fine-tuning...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    print("Wrapped model with LoRA adapters.")
    model.print_trainable_parameters()

    # --- Initialize AutoProcessor ---
    print("Initializing AutoProcessor...")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            if hasattr(model, 'config') and model.config.pad_token_id is None:
                model.config.pad_token_id = processor.tokenizer.pad_token_id
            print(f"AutoProcessor's tokenizer pad_token set to: {processor.tokenizer.pad_token}")
        print("AutoProcessor initialized.")
    except Exception as e:
        print(f"\nError initializing AutoProcessor: {e}")
        exit()

    train_dataset = ImageCaptioningDataset(
        annotations_file=CAPTION_ANNOTATIONS_FILE,
        image_folder=IMAGE_FOLDER,
        image_col=IMAGE_COL_NAME,
        target_caption_cols=TARGET_CAPTION_COLUMNS,
        split_col=SPLIT_COL_NAME,
        target_split=TRAIN_TARGET_SPLIT,
        max_samples=MAX_TRAIN_SAMPLES
    )
    collate_wrapper = lambda batch: collate_fn_captioning(batch, processor, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_wrapper,
        num_workers=2
    )
    print(f"Training DataLoader prepared with {len(train_dataset)} samples.")

    val_dataset = ImageCaptioningDataset(
        annotations_file=CAPTION_ANNOTATIONS_FILE,
        image_folder=IMAGE_FOLDER,
        image_col=IMAGE_COL_NAME,
        target_caption_cols=TARGET_CAPTION_COLUMNS,
        split_col=SPLIT_COL_NAME,
        target_split=VAL_TARGET_SPLIT,
        max_samples=MAX_VAL_SAMPLES
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_wrapper,
        num_workers=2
    )
    print(f"Validation DataLoader prepared with {len(val_dataset)} samples.")

    print("Setting up optimizer for LoRA parameters...")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    print(f"Optimizer: AdamW, Learning Rate: {LEARNING_RATE}")

    print(f"Starting decoder-only LoRA training for {NUM_EPOCHS} epochs on {len(train_dataset)} training samples...")
    global_step_for_wandb = 0
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} (Training) ---")
        epoch_train_loss_sum = 0.0
        steps_in_epoch_train = 0
        progress_bar_train = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Train", leave=False)

        for batch_data in progress_bar_train:
            if batch_data is None: continue
            try:
                pixel_values = batch_data['pixel_values'].to(device)
                input_ids = batch_data['input_ids'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                labels = batch_data['labels'].to(device)
                if device.type == 'cuda':
                    pixel_values = pixel_values.to(model_dtype)
            except Exception as e:
                continue

            optimizer.zero_grad()
            try:
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs['logits']
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            except Exception as e:
                print(f"\nError during forward/loss (train): {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            try:
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"\nError during backward/step (train): {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            current_loss = loss.item()
            epoch_train_loss_sum += current_loss
            global_step_for_wandb += 1
            steps_in_epoch_train += 1
            progress_bar_train.set_postfix({'loss': f"{current_loss:.4f}"})
            wandb.log({"train/step_loss": current_loss, "trainer/global_step": global_step_for_wandb})

            # Clear cache periodically
            if global_step_for_wandb % 100 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_epoch_train_loss = epoch_train_loss_sum / steps_in_epoch_train if steps_in_epoch_train > 0 else 0
        print(f"--- End of Epoch {epoch + 1} (Training) ---")
        print(f"Average Training Loss: {avg_epoch_train_loss:.4f}")
        wandb.log({"train/epoch_avg_loss": avg_epoch_train_loss, "epoch": epoch + 1})

        model.eval()
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} (Validation) ---")
        epoch_val_loss_sum = 0.0
        steps_in_epoch_val = 0
        progress_bar_val = tqdm(val_dataloader, desc=f"Epoch {epoch + 1} Val", leave=False)
        with torch.no_grad():
            for batch_data in progress_bar_val:
                if batch_data is None: continue
                try:
                    pixel_values = batch_data['pixel_values'].to(device)
                    input_ids = batch_data['input_ids'].to(device)
                    attention_mask = batch_data['attention_mask'].to(device)
                    labels = batch_data['labels'].to(device)
                    if device.type == 'cuda':
                        pixel_values = pixel_values.to(model_dtype)
                except Exception as e:
                    continue
                try:
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs['logits']
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                except Exception as e:
                    print(f"\nError during forward/loss (val): {e}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                current_val_loss = loss.item()
                epoch_val_loss_sum += current_val_loss
                steps_in_epoch_val += 1
                progress_bar_val.set_postfix({'val_loss': f"{current_val_loss:.4f}"})
        avg_epoch_val_loss = epoch_val_loss_sum / steps_in_epoch_val if steps_in_epoch_val > 0 else float('inf')
        print(f"--- End of Epoch {epoch + 1} (Validation) ---")
        print(f"Average Validation Loss: {avg_epoch_val_loss:.4f}")
        wandb.log({"val/epoch_avg_loss": avg_epoch_val_loss, "epoch": epoch + 1})

        # Test generation on a sample to check for repetitive tokens
        if epoch + 1 == NUM_EPOCHS or epoch % 1 == 0:  # Test on last epoch and every epoch
            print("\n--- Testing generation for repetitive tokens ---")
            try:
                # Select a random sample from validation
                test_idx = random.randint(0, len(val_dataset) - 1)
                test_sample = val_dataset[test_idx]
                if test_sample is not None:
                    test_image, test_prompt, test_target = test_sample
                    # Process for generation
                    test_inputs = processor(
                        text=[test_prompt],
                        images=test_image,
                        return_tensors="pt"
                    )
                    # Move to device
                    for k, v in test_inputs.items():
                        if isinstance(v, torch.Tensor):
                            test_inputs[k] = v.to(device)
                    if device.type == 'cuda':
                        test_inputs['pixel_values'] = test_inputs['pixel_values'].to(model_dtype)

                    # Generate caption
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **test_inputs,
                            max_new_tokens=MAX_TARGET_CAPTION_LENGTH // 4,
                            min_new_tokens=5,
                            do_sample=False,
                            num_beams=4,
                            repetition_penalty=2.2,  # Increased to prevent repetitions
                            length_penalty=0.8,
                            early_stopping=True
                        )

                    # Decode
                    input_length = test_inputs['input_ids'].shape[1]
                    generated_text = tokenizer.decode(
                        generated_ids[0][input_length:],
                        skip_special_tokens=True
                    )

                    # Also decode without skipping special tokens to check for problems
                    generated_with_special = tokenizer.decode(
                        generated_ids[0][input_length:],
                        skip_special_tokens=False
                    )

                    # Print results
                    print(f"Ground truth: {test_target}")
                    print(f"Generated: {generated_text}")
                    if "<seg" in generated_with_special or "<loc" in generated_with_special:
                        print(f"WARNING: Special tokens detected: {generated_with_special}")

                    # Log to wandb
                    wandb.log({
                        f"examples/test_ground_truth": test_target,
                        f"examples/test_generated": generated_text,
                        "epoch": epoch + 1
                    })
            except Exception as e:
                print(f"Error during test generation: {e}")

        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model for epoch {epoch + 1}...")
            best_epoch_output_dir = os.path.join(OUTPUT_DIR, "best_model")
            os.makedirs(best_epoch_output_dir, exist_ok=True)
            model.save_pretrained(best_epoch_output_dir)
            tokenizer.save_pretrained(best_epoch_output_dir)
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch + 1
            print("Best model checkpoint saved.")

    print("\nTraining finished.")
    print(f"Saving final LoRA adapters to {OUTPUT_DIR} (last epoch)...")
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    print("Final adapter saving complete.")
    print(f"Fine-tuned LoRA adapters saved in: {OUTPUT_DIR}")
    wandb.finish()


if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    main()
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("Script finished and cleanup attempted.")