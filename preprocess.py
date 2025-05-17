import pandas as pd
import re
import os
import nltk
from nltk.tokenize import word_tokenize
import string
from tqdm import tqdm

# Configuration
INPUT_CSV = '../dataset/RISCM/captions.csv'  # Your current CSV file
OUTPUT_CSV = '../dataset/RISCM/processed_captions.csv'  # Where to save the processed CSV
IMAGE_FOLDER = '../dataset/RISCM/resized'  # Path to verify images exist
MIN_WORDS = 5  # Minimum word count to keep caption
MAX_WORDS = 30  # Maximum word count

# Try to download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    print("NLTK download failed, but we can continue")


# Function to preprocess a caption
def preprocess_caption(caption):
    if not isinstance(caption, str) or pd.isna(caption):
        return ""

    # Remove quotes and irregular formatting
    caption = caption.replace('"', '').replace('"', '').replace('"', '')

    # Fix spacing issues
    caption = re.sub(r'\s+', ' ', caption)
    caption = re.sub(r'\s+\.', '.', caption)
    caption = re.sub(r'\s+,', ',', caption)

    # Fix capitalization after commas
    caption = re.sub(r',\s*([A-Z])', lambda m: ', ' + m.group(1).lower(), caption)

    # Ensure consistent capitalization for sentence start
    caption = caption.strip()
    if caption and caption[0].islower():
        caption = caption[0].upper() + caption[1:]

    # Ensure period at end
    if caption and not caption.endswith('.'):
        caption += '.'

    # Remove double periods and fix other punctuation issues
    caption = caption.replace('..', '.')
    caption = caption.replace(',.', '.')

    return caption.strip()


# Function to check caption quality (returns True if good quality)
def is_good_quality(caption):
    if not caption:
        return False

    # Count words (after tokenization for accuracy)
    words = word_tokenize(caption.lower())
    word_count = len([w for w in words if w not in string.punctuation])

    # Check if length is appropriate
    if word_count < MIN_WORDS or word_count > MAX_WORDS:
        return False

    # Check for specific low-information patterns
    low_info_patterns = [
        r'^There (is|was) (a|an|one) (plane|airplane)( in the parking lot| on the ground| on the runway)?\.?$',
        r'^One plane is on the ground\.?$',
        r'^A plane is on the ground\.?$',
        r'^The plane is on the ground\.?$',
        r'^The buildings are next to an airplane\.?$',
        r'^(A|An) plane is on the ground\.?$',
        r'^(A|One) plane is in the parking lot\.?$',
        r'^There is a plane on the runway\.?$',
        r'^There is a plane beside the grass\.?$'
    ]

    for pattern in low_info_patterns:
        if re.match(pattern, caption, re.IGNORECASE):
            return False

    return True


def main():
    print("Starting caption preprocessing...")

    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input CSV file {INPUT_CSV} not found.")
        return

    # Read the input CSV
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Read {len(df)} rows from {INPUT_CSV}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Make sure we have the expected columns
    expected_columns = ['source', 'split', 'image', 'caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']
    for col in expected_columns:
        if col not in df.columns:
            print(f"Warning: Expected column '{col}' not found in CSV")

    # Process each row
    processed_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing captions"):
        # Skip if image doesn't exist (optional - you can comment this out if you don't want to check)
        image_path = os.path.join(IMAGE_FOLDER, row['image'])
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        # Process each caption column
        new_row = {'source': row['source'], 'split': row['split'], 'image': row['image']}
        valid_captions = []

        for i in range(1, 6):
            caption_col = f'caption_{i}'
            if caption_col in row and pd.notna(row[caption_col]):
                clean_caption = preprocess_caption(row[caption_col])
                if is_good_quality(clean_caption):
                    valid_captions.append(clean_caption)

        # Only include rows with at least one valid caption
        if valid_captions:
            # Add valid captions to new row
            for i, caption in enumerate(valid_captions, 1):
                new_row[f'caption_{i}'] = caption

            # Fill remaining caption columns with empty strings
            for i in range(len(valid_captions) + 1, 6):
                new_row[f'caption_{i}'] = ""

            processed_rows.append(new_row)

    # Create the processed DataFrame
    processed_df = pd.DataFrame(processed_rows)

    # Save to CSV
    processed_df.to_csv(OUTPUT_CSV, index=False)

    # Print statistics
    print("\nPreprocessing complete!")
    print(f"Original entries: {len(df)}")
    print(f"Processed entries: {len(processed_df)}")

    # Count valid captions before and after
    original_captions = 0
    for i in range(1, 6):
        col = f'caption_{i}'
        if col in df.columns:
            original_captions += df[col].notna().sum()

    processed_captions = 0
    for i in range(1, 6):
        col = f'caption_{i}'
        if col in processed_df.columns:
            processed_captions += (processed_df[col] != "").sum()

    print(f"Original captions: {original_captions}")
    print(f"Processed captions: {processed_captions}")
    print(f"Captions filtered: {original_captions - processed_captions}")

    # Print some examples
    print("\nSample processed entries:")
    for i in range(min(3, len(processed_df))):
        row = processed_df.iloc[i]
        print(f"Image: {row['image']}")
        for j in range(1, 6):
            caption_col = f'caption_{j}'
            if caption_col in row and row[caption_col]:
                print(f"  Caption {j}: {row[caption_col]}")
        print()

    print(f"Processed data saved to {OUTPUT_CSV}")
    print("You can now use this file for fine-tuning!")


if __name__ == "__main__":
    main()