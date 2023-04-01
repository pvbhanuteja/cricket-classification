import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from dataset import CustomDataset
from preprocess import SR
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

class CricketClassifier(LightningModule):
    def __init__(self, label2id, id2label, num_classes):
        super().__init__()
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", label2id=label2id, id2label=id2label ,num_labels=num_classes,ignore_mismatched_sizes=True)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def log_gradients(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                self.logger.experiment.add_histogram(
                    f"Gradients/{name}", param.grad, self.global_step)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        logits = outputs.logits
        loss = self.criterion(logits, labels)

        # Log and print gradients
        self.log_gradients()
        self.log('train_loss', loss)
        print(f"Train Loss: {loss:.4f}", end="\r")
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        logits = outputs.logits
        _, predicted = torch.max(logits, dim=1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        self.log('Test Accuracy', accuracy)
        print(f"Test Accuracy: {accuracy:.4f}", end="\r")
        return accuracy

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

# Rest of the code remains the same


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

# Initialize the LightningModule
classifier = CricketClassifier(label2id, id2label, num_classes)

# Set up TensorBoard logger
logger = TensorBoardLogger("lightning_logs", name="cricket_experiment")

# Initialize the Trainer
trainer = Trainer(
    max_epochs=100,
    devices=2, 
    accelerator="gpu",
    strategy="ddp",
    logger=logger,
    gradient_clip_val=1.0
)
# Train the model
trainer.fit(classifier, DataLoader(train_set, batch_size=8, shuffle=True), DataLoader(test_set, batch_size=8, shuffle=False))

# Save the best model
model_save_path = f"{logger.log_dir}/best_finetuned_ast_cricket_data.pt"
torch.save(classifier.model.state_dict(), model_save_path)
