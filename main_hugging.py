import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import CustomDataset
from preprocess import SR
from transformers import (
    AutoFeatureExtractor, ASTForAudioClassification,
    TrainingArguments, Trainer
)
from tqdm import tqdm

# Custom dataset class
class HFTrainerDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, idx):
        return {"input_values": self.dataset[idx][0], "labels": self.dataset[idx][1]}
    
    def __len__(self):
        return len(self.dataset)

# Load the dataset
dataset = CustomDataset(data_path='cricket_data_feature_extracted.pt', type='main')
generator = torch.Generator().manual_seed(42)

label2id = dataset.label2id
id2label = dataset.id2label

num_classes = len(label2id)

# Set the train-test split ratio
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset into train and test sets
train_set, test_set = random_split(
    dataset, [train_size, test_size], generator=generator)

# Convert the datasets to be compatible with Hugging Face Trainer
hf_train_set = HFTrainerDataset(train_set)
hf_test_set = HFTrainerDataset(test_set)

# Initialize the model
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593", label2id=label2id, id2label=id2label ,num_labels=num_classes,ignore_mismatched_sizes=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="hf_trainer_output",
    logging_dir="hf_trainer_logs",
    num_train_epochs=10,
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

# Define the compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).sum().item()
    total = len(labels)
    accuracy = correct / total
    return {"accuracy": accuracy}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_train_set,
    eval_dataset=hf_test_set,
    compute_metrics=compute_metrics,
)

# Train and evaluate the model
trainer.train()
trainer.evaluate()

# Save the best model
model_save_path = f"{training_args.output_dir}/best_finetuned_ast_cricket_data.pt"
model.save_pretrained(model_save_path)
print(f"Best Model saved to {model_save_path}")
