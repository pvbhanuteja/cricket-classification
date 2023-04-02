import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from dataset import CustomDataset
from preprocess import SR
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# Load the dataset
dataset = CustomDataset(data_path='cricket_data_feature_extracted.pt', type='main')
generator = torch.Generator().manual_seed(42)

label2id = dataset.label2id
id2label = dataset.id2label

num_classes = len(label2id)

print("Number of classes: ", num_classes)
# Set the train-test split ratio
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset into train and test sets
train_set, test_set = random_split(
    dataset, [train_size, test_size], generator=generator)
batch_size = 2
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Initialize the model and feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593", label2id=label2id, id2label=id2label ,num_labels=num_classes,ignore_mismatched_sizes=True)

# Set up the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the model
num_epochs = 10
best_test_acc = 0.0
# Create TensorBoard SummaryWriter
writer = SummaryWriter("runs/cricket_experiment")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for idx, (inputs, labels) in enumerate(train_loader):
    
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        logits = outputs.logits

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}", end="\r")
    # Calculate average training loss
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}")
    
    # Log training loss to TensorBoard
    writer.add_scalar("Training Loss", avg_train_loss, epoch)
    
    # Evaluate the model on the test set
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    
    # Log test accuracy to TensorBoard
    writer.add_scalar("Test Accuracy", accuracy, epoch)
    print(f"Test Accuracy: {accuracy:.4f} Epoch [{epoch + 1}/{num_epochs}]")
    
    if accuracy > best_test_acc:
        best_test_acc = accuracy
        model_save_path = f"{writer.log_dir}/best_finetuned_ast_cricket_data.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Best Model saved to {model_save_path}")

# Close TensorBoard SummaryWriter
writer.close()
