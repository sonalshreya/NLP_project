"""
Hypothesis-Only Baseline
Trains model on hypothesis only to detect dataset biases
"""

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
import json
import torch
import os

def train_hypothesis_only(checkpoint_dir=None):
    """Train model on hypothesis only"""
    
    print("="*70)
    print("HYPOTHESIS-ONLY BASELINE")
    print("="*70)
    print("\nThis model sees ONLY the hypothesis (no premise!)")
    print("If it performs well, it proves the dataset has biases.\n")
    
    save_dir = checkpoint_dir if checkpoint_dir else './outputs/hypothesis_only'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    print("1. Loading SNLI dataset...")
    dataset = load_dataset('snli')
    dataset = dataset.filter(lambda x: x['label'] != -1)
    
    print(f"   Train: {len(dataset['train']):,}")
    print(f"   Val:   {len(dataset['validation']):,}")
    print(f"   Test:  {len(dataset['test']):,}")
    
    # Load model
    print("\n2. Loading ELECTRA-small-discriminator...")
    model_name = 'google/electra-small-discriminator'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize ONLY hypothesis (ignore premise!)
    print("\n3. Tokenizing HYPOTHESIS ONLY (ignoring premise)...")
    def tokenize_hypothesis_only(examples):
        return tokenizer(
            examples['hypothesis'],  # ONLY hypothesis!
            truncation=True,
            padding='max_length',
            max_length=128
        )
    
    tokenized = dataset.map(tokenize_hypothesis_only, batched=True)
    tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Training args
    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'accuracy': (predictions == labels).mean()}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n4. Training hypothesis-only model...")
    print("="*70)
    trainer.train()
    
    # Evaluate
    print("\n5. Evaluating on test set...")
    test_results = trainer.evaluate(tokenized['test'])
    
    print("\n" + "="*70)
    print("HYPOTHESIS-ONLY RESULTS")
    print("="*70)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Expected:      0.65-0.70")
    print("="*70)
    
    if test_results['eval_accuracy'] > 0.60:
        print("\n⚠️  HIGH ACCURACY WITH HYPOTHESIS ONLY!")
        print("   This PROVES the dataset has significant biases!")
        print(f"   The model achieves {test_results['eval_accuracy']*100:.1f}% without seeing the premise!")
    
    # Save results
    os.makedirs('./analysis/results', exist_ok=True)
    with open('./analysis/results/hypothesis_only_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Model saved to: {save_dir}")
    
    return trainer, test_results

if __name__ == "__main__":
    trainer, results = train_hypothesis_only()
