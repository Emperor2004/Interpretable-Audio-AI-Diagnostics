import os
import torch
import evaluate
import numpy as np
from datasets import load_dataset, DatasetDict, Audio  # 1. IMPORT Audio
import librosa  # 2. IMPORT librosa
import io       # 3. IMPORT io
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    TrainingArguments,
    Trainer,
)

# --- 1. CONFIGURATION ---
MODEL_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
DATASET_NAME = "ashraq/esc50"
NEW_MODEL_DIR = "./my-finetuned-model" 

# Training settings
NUM_EPOCHS = 1
BATCH_SIZE = 8  # If you get a CUDA Out of Memory error, change this to 8.

print(f"Starting fine-tuning process...")
print(f"Base Model: {MODEL_CHECKPOINT}")
print(f"Dataset: {DATASET_NAME}")

# --- 2. LOAD DATASET & PROCESSOR ---
print("Loading dataset and feature extractor...")

# Load the dataset. We will NOT decode audio here.
dataset = load_dataset(DATASET_NAME) 

# --- START FIX ---
# 4. Cast the 'audio' column to Audio(decode=False).
# This tells datasets: "Don't try to decode. Give me the raw data."
# This is the step that provides the {'path': None, 'bytes': b'...'} object.
print("Casting audio column to return raw data (decode=False)...")
dataset = dataset.cast_column("audio", Audio(decode=False))
# --- END FIX ---

# Load the AST feature extractor
feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)

# Get the number of labels
label_col = "category"
print("Extracting unique labels from the 'category' column...")
labels = dataset["train"].unique(label_col)
labels.sort() 

num_labels = len(labels)
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

print(f"Found {num_labels} unique labels. First 5: {labels[:5]}")

# --- 3. PRE-PROCESSING FUNCTION ---
def preprocess_function(examples):
    """
    Manually loads and processes raw audio *from bytes* using librosa
    into model inputs (Mel Spectrograms).
    """
    audio_arrays = []
    
    # --- START FIX ---
    # 5. 'examples["audio"]' is now a list of dicts {'path': None, 'bytes': b'...'}
    #    We load each file's BYTES using librosa and io.BytesIO
    for audio_info in examples["audio"]:
        if audio_info["bytes"]:
            # Wrap the bytes in a file-like object for librosa
            audio_bytes = io.BytesIO(audio_info["bytes"])
            waveform, sr = librosa.load(audio_bytes, sr=16000, mono=True)
            audio_arrays.append(waveform)
        elif audio_info["path"]:
            # Fallback just in case, though we expect bytes
            waveform, sr = librosa.load(audio_info["path"], sr=16000, mono=True)
            audio_arrays.append(waveform)
    # --- END FIX ---
    
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=16000, # Explicitly set to 16kHz
        return_tensors="pt"
    )
    inputs["labels"] = [label2id[example] for example in examples[label_col]]
    return inputs

print("Applying preprocessing to the dataset (this will take a few minutes)...")
# This will now work, as preprocess_function loads from bytes
encoded_dataset = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=["audio", "esc10"] # Remove old columns after processing
)


# --- 4. SPLIT DATASET ---
print("Splitting single 'train' set into train/validation/test...")
train_val_split = encoded_dataset["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
train_test_split = train_val_split['train'].train_test_split(test_size=0.1, shuffle=True, seed=42)

final_dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test'], 
    'test': train_val_split['test']        
})

print("Dataset splits finalized:")
print(final_dataset)


# --- 5. LOAD MODEL ---
print("Loading pre-trained model for fine-tuning...")
model = ASTForAudioClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels, 
    label2id=label2id,     
    id2label=id2label,
    ignore_mismatched_sizes=True 
)

# --- 6. DEFINE TRAINING ---
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy during evaluation."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)

training_args = TrainingArguments(
    output_dir=f"{NEW_MODEL_DIR}-checkpoints",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=NUM_EPOCHS,
    fp16=torch.cuda.is_available(), 
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    # load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    resume_from_checkpoint=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["validation"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

# --- 7. TRAIN AND EVALUATE ---
print("\n*** STARTING TRAINING ***\n")
trainer.train()
print("\n*** TRAINING COMPLETE ***\n")

print("Evaluating final model on test set...")
eval_results = trainer.evaluate(final_dataset["test"])

print("\n*** TEST SET EVALUATION RESULTS ***\n")
print(eval_results)

# --- 8. SAVE YOUR NEW MODEL ---
print(f"Saving fine-tuned model to {NEW_MODEL_DIR}...")
trainer.save_model(NEW_MODEL_DIR)
feature_extractor.save_pretrained(NEW_MODEL_DIR) 

print("\n*** ALL DONE. ***")
print(f"Your fine-tuned model is saved in the '{NEW_MODEL_DIR}' folder.")
print(f"You can now update 'backend/app/model_loader.py' to use this model.")