import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
from tqdm.auto import tqdm
import gc
import os
import re

INPUT_CSV_PATH = '../dataset/RISCM/captions.csv'
OUTPUT_CSV_PATH = '../dataset_with_steps_batched.csv'
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
NUM_ROWS_TO_PROCESS = None
CAPTION_COLUMN_NAMES = ['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']
SPLIT_COL_NAME = 'split'
IMAGE_COL_NAME = 'image'

LOAD_IN_4BIT = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 250
TEMPERATURE = 0.6
TOP_P = 0.9
DO_SAMPLE = True
BATCH_SIZE = 16

# --- Load Model and Tokenizer ---
print(f"Using device: {DEVICE}")
if LOAD_IN_4BIT and DEVICE == "cpu":
    LOAD_IN_4BIT = False
    print("Disabled 4-bit loading for CPU.")

quantization_config = None
if LOAD_IN_4BIT:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("Using 4-bit quantization.")

print(f"Loading tokenizer for {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side='left')
print(f"Loading model {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config if LOAD_IN_4BIT else None,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Llama 3 needs a pad token for batching
    print(f"Set tokenizer.pad_token to {tokenizer.pad_token} (EOS token)")
    if hasattr(model, 'config') and model.config.pad_token_id is None:
         model.config.pad_token_id = model.config.eos_token_id


# --- Function to Generate Steps for a Batch ---
def generate_steps_batched(captions_batch, model, tokenizer):
    if not captions_batch:
        return []

    system_prompt = """Decompose the user-provided caption into logical steps. Output *only* the steps starting with "Step 1:" and ending *exactly* with the line 'Final Caption: [Original Caption]'. Do not include any conversational text, greetings, or explanations."""

    # Prepare a list of message lists, one for each caption in the batch
    batch_messages = []
    for caption_text in captions_batch:
        if not caption_text or not isinstance(caption_text, str) or len(caption_text.strip()) == 0:
            # Add a placeholder error for invalid captions in batch
            batch_messages.append([{"role": "system", "content": "Error: Invalid caption input."}])
            continue
        user_content = f"""Caption: "{caption_text.strip()}" """
        batch_messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ])

    formatted_prompts = []
    valid_captions_in_batch = []
    for i, messages in enumerate(batch_messages):
        if "Error: Invalid caption input." in messages[0]["content"]:
            formatted_prompts.append("Error: Invalid caption input.") # Keep placeholder
            valid_captions_in_batch.append(None)
        else:
            try:
                formatted_prompts.append(
                    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                )
                valid_captions_in_batch.append(captions_batch[i]) # Store original caption for validation
            except Exception as e:
                print(f"Error applying chat template for a caption: {e}")
                formatted_prompts.append(f"Error: Chat template application failed - {e}")
                valid_captions_in_batch.append(None)


    # Tokenize the batch of formatted prompts
    try:
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings - MAX_NEW_TOKENS - 10
        ).to(model.device)
    except Exception as e:
        return [f"Error: Tokenization failed - {e}" for _ in captions_batch]


    generated_steps_batch = ["Error: Generation failed" for _ in captions_batch]
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, # Pass tokenized batch
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode batch of responses
        # outputs will contain input_ids as well, slice to get only generated tokens
        decoded_responses = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        final_results = []
        for i, response_text in enumerate(decoded_responses):
            original_caption = valid_captions_in_batch[i]
            if original_caption is None or "Error:" in formatted_prompts[i]: # If input was invalid or template failed
                final_results.append(formatted_prompts[i]) # Propagate error
                continue

            cleaned_steps = re.sub(r'\n\s*\n+', '\n', response_text.strip()).strip()

            # Basic validation (optional)
            if not cleaned_steps.startswith("Step 1:") or f"Final Caption: {original_caption.strip()}" not in cleaned_steps:
                # print(f"Warning: Output format may be incorrect for '{original_caption[:20]}...'")
                pass # Keep minimal
            final_results.append(cleaned_steps)
        generated_steps_batch = final_results

    except Exception as e:
        print(f"Error during batched generation: {e}")
        generated_steps_batch = [f"Error: Generation exception - {e}" for _ in captions_batch]
    finally:
        del inputs
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return generated_steps_batch


# --- Main Processing ---
if not os.path.exists(INPUT_CSV_PATH):
     print(f"Error: Input file not found at {INPUT_CSV_PATH}")
     exit()

df_full = pd.read_csv(INPUT_CSV_PATH)
print(f"Read {len(df_full)} rows from {INPUT_CSV_PATH}.")

if NUM_ROWS_TO_PROCESS is not None and NUM_ROWS_TO_PROCESS > 0 and NUM_ROWS_TO_PROCESS < len(df_full):
    df_process = df_full.head(NUM_ROWS_TO_PROCESS).copy()
    print(f"Processing first {NUM_ROWS_TO_PROCESS} image rows.")
elif NUM_ROWS_TO_PROCESS is None or NUM_ROWS_TO_PROCESS == 0:
     df_process = df_full
     print(f"Processing all {len(df_full)} image rows.")
else: # NUM_ROWS_TO_PROCESS >= len(df_full)
     df_process = df_full
     print(f"Processing all {len(df_full)} image rows (NUM_ROWS_TO_PROCESS >= total rows).")


required_columns = [SPLIT_COL_NAME, IMAGE_COL_NAME] + CAPTION_COLUMN_NAMES
missing_cols = [col for col in required_columns if col not in df_process.columns]
if missing_cols:
    print(f"Error: Missing required columns in CSV: {missing_cols}")
    exit()

output_data = []
captions_buffer = [] # Buffer for batching
metadata_buffer = [] # To store (split_info, image_info, original_caption)

total_captions_to_iterate = len(df_process) * len(CAPTION_COLUMN_NAMES)
pbar = tqdm(total=total_captions_to_iterate, desc="Preparing Captions")

for index, row in df_process.iterrows():
    try:
        split_info = row[SPLIT_COL_NAME]
        image_info = row[IMAGE_COL_NAME]
    except KeyError:
        pbar.update(len(CAPTION_COLUMN_NAMES)) # Skip all captions for this row
        continue

    for caption_col_name in CAPTION_COLUMN_NAMES:
        pbar.update(1)
        try:
            caption = row[caption_col_name]
            if pd.isna(caption) or not isinstance(caption, str) or len(caption.strip()) == 0 :
                 continue
        except KeyError:
             continue

        captions_buffer.append(caption)
        metadata_buffer.append({'split': split_info, 'image': image_info, 'original_caption': caption})

        if len(captions_buffer) >= BATCH_SIZE:
            pbar.set_description(f"Generating CoT for batch of {len(captions_buffer)}")
            generated_cots_batch = generate_steps_batched(captions_buffer, model, tokenizer)
            for i, cot_response_text in enumerate(generated_cots_batch):
                meta = metadata_buffer[i]
                output_data.append({
                    'split': meta['split'],
                    'image': meta['image'],
                    'original_caption': meta['original_caption'],
                    'cot_response': cot_response_text
                })
            captions_buffer = []
            metadata_buffer = []
pbar.close()

# Process any remaining captions in the buffer
if captions_buffer:
    print(f"Processing remaining {len(captions_buffer)} captions...")
    generated_cots_batch = generate_steps_batched(captions_buffer, model, tokenizer)
    for i, cot_response_text in enumerate(generated_cots_batch):
        meta = metadata_buffer[i]
        output_data.append({
            'split': meta['split'],
            'image': meta['image'],
            'original_caption': meta['original_caption'],
            'cot_response': cot_response_text
        })

output_df = pd.DataFrame(output_data)
output_df.to_csv(OUTPUT_CSV_PATH, index=False, header=True)
print(f"\nProcessing complete. Output saved to: {OUTPUT_CSV_PATH} ({len(output_df)} CoT entries generated)")

# --- Cleanup ---
del model
del tokenizer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
print("Cleanup complete.")
