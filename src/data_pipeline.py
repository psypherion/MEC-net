import os
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# --- 1. Define Constants and Tokenizer ---
# We use a small, pre-trained model for the tokenizer.
MODEL_NAME = "distilbert-base-uncased"
# This is the "episode" size. We'll chunk the text into blocks of this size.
BLOCK_SIZE = 128
# Small batch size to manage memory on local hardware.
BATCH_SIZE = 4
# Path to save the processed dataset
PROCESSED_DATA_PATH = "./data/wikitext-2-processed"

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer loaded: {MODEL_NAME}")

# --- 2. Load or Process the Dataset ---
# Check if a processed version of the dataset already exists.
if os.path.exists(PROCESSED_DATA_PATH):
    print("Found processed dataset. Loading from disk...")
    lm_datasets = load_from_disk(PROCESSED_DATA_PATH)
    print("Dataset loaded successfully from disk.")
else:
    print("Processed dataset not found. Starting data pipeline...")
    try:
        # We'll use the non-streaming version for the full processing pipeline.
        raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()  # Exit if we can't load the raw dataset


    # --- 3. Tokenize the Text ---
    # This function tokenizes the text from the dataset.
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, padding=True)


    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    print("Text tokenized successfully.")


    # --- 4. Chunk the Tokenized Text into Episodes ---
    # This is the core logic for creating our "episodes" or data chunks.
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

        result = {
            k: [t[i: i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated_examples.items()
        }
        return result


    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )
    print("Dataset chunked into episodes of size:", BLOCK_SIZE)

    # --- 5. Save the Processed Dataset ---
    print(f"Saving processed dataset to {PROCESSED_DATA_PATH}...")
    lm_datasets.save_to_disk(PROCESSED_DATA_PATH)
    print("Dataset saved successfully.")

# --- 6. Create a DataLoader for Batching ---
lm_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
dataloader = DataLoader(lm_datasets, batch_size=BATCH_SIZE)
print("DataLoader created successfully.")

# --- 7. Demonstrate the Pipeline Output ---
print("\nProcessing a few batches of episodes...")
for i, batch in enumerate(dataloader):
    if i >= 3:
        break

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    print(f"\nBatch {i + 1}:")
    print("Input IDs shape:", input_ids.shape)
    print("Attention Mask shape:", attention_mask.shape)

    assert input_ids.shape[0] == BATCH_SIZE
    assert input_ids.shape[1] == BLOCK_SIZE

print("\nData pipeline demonstration complete. The dataloader is ready for the MEC-Net++ 'wake' phase.")
