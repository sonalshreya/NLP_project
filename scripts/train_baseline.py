"""
Baseline Model Training Script
Train ELECTRA-small on SNLI dataset
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

def train_baseline(output_dir='./outputs/baseline', checkpoint_dir=None):
    """
    Train baseline ELECTRA-small model on SNLI
    
    Args:
        output_dir: Where to save model in repo (for structure)
        checkpoint_dir: Where to save actual checkpoints (Google Drive)
    """
    
    print("="*70)
    print("TRAINING BASELINE MODEL ON SNLI")
    print("="*70)
    
    # Use checkpoint_dir if provided (for Drive), otherwise output_dir
    save_dir = checkpoint_dir if checkpoint_dir else output_dir
    
    # Load SNLI dataset
    print("\n1. Loading SNLI dataset...")
    dataset = load_dataset('snli')
    dataset = dataset.filter(lambda x: x['label'] != -1)
    
    print(f"   Train: {len(dataset['train']):,}")
    print(f"   Validation: {len(dataset['validation']):,}")
    print(f"   Test: {len(dataset['test']):,}")
    
    # Load model
    print("\n2. Loading ELECTRA-small-discriminator...")
    model_name = 'google/electra-small-discriminator'
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize
    print("\n3. Tokenizing...")
    def tokenize(examples):
        return tokenizer(
            examples['premise'],
            examples['hypothesis'],
            truncation=True,
            padding='max_length',
            max_length=128
        )
    
    tokenized = dataset.map(tokenize, batched=True, desc="Tokenizing")
    tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Training arguments
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
        metric_for_best_model='accuracy',
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # Metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {'accuracy': accuracy}
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n4. Training...")
    print("="*70)
    trainer.train()
    
    # Evaluate
    print("\n5. Evaluating on test set...")
    test_results = trainer.evaluate(tokenized['test'])
    
    print("\n" + "="*70)
    print("BASELINE RESULTS")
    print("="*70)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Expected: 0.88-0.90")
    print("="*70)
    
    # Save results
    results_file = output_dir.replace('outputs', 'analysis/results') + '_results.json'
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✓ Model saved to: {save_dir}")
    print(f"✓ Results saved to: {results_file}")
    
    return trainer, test_results

if __name__ == "__main__":
    # For Drive persistence, pass checkpoint_dir
    # trainer, results = train_baseline(
    #     output_dir='./outputs/baseline',
    #     checkpoint_dir='/content/drive/MyDrive/nlp-final-project/checkpoints/baseline'
    # )
    
    trainer, results = train_baseline()
