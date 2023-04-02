import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from dataset import CustomDataset
from preprocess import SR
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

class CricketClassifier(LightningModule):
    def __init__(self, label2id, id2label, num_classes):
        super().__init__()
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", label2id=label2id, id2label=id2label ,num_labels=num_classes,ignore_mismatched_sizes=True)
        self.criterion = nn.CrossEntropyLoss()
        self.training_step_outputs = []
        self.val_step_outputs = []

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
        # self.log_gradients()
        metrics = {"train_loss_step": loss.item()}
        self.logger.log_metrics(metrics, step=self.global_step)
        train_outputs = {"loss": loss, "labels": labels, "predictions": logits}
        self.training_step_outputs.append(train_outputs)
        # print(f"Train Loss (step): {loss:.4f}", end="\r")
        return train_outputs

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics = {"train_loss_epoch": avg_loss.item()}
        self.logger.log_metrics(metrics, step=self.current_epoch)
        
        # Log labels and predictions of the last step in the current epoch
        last_step_outputs = outputs[-1]
        last_step_labels = last_step_outputs["labels"].tolist()
        last_step_predictions = torch.argmax(last_step_outputs["predictions"], dim=1).tolist()

        # Log the last_step_labels and last_step_predictions as text
        last_step_labels_str = ', '.join(map(str, last_step_labels))
        last_step_predictions_str = ', '.join(map(str, last_step_predictions))


        self.logger.experiment.add_text(f"Last step labels", last_step_labels_str, global_step=self.current_epoch)
        self.logger.experiment.add_text(f"Last step predictions", last_step_predictions_str, global_step=self.current_epoch)
        
        print(f"Train Loss (epoch): {avg_loss:.4f}")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        logits = outputs.logits
        _, predicted = torch.max(logits, dim=1)
        val_outputs = {"labels": labels, "predictions": predicted}
        self.val_step_outputs.append(val_outputs)
        # Return values to be used in validation_epoch_end
        return val_outputs
    
    def on_validation_epoch_end(self):
        outputs = self.val_step_outputs
        all_labels = torch.cat([x['labels'] for x in outputs], dim=0)
        all_predictions = torch.cat([x['predictions'] for x in outputs], dim=0)

        # Calculate accuracy
        accuracy = (all_predictions == all_labels).sum().item() / all_labels.size(0)

        # Calculate precision, recall, F1-score, and confusion matrix
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels.cpu(), all_predictions.cpu(), average='macro')
        conf_mat = confusion_matrix(all_labels.cpu(), all_predictions.cpu())

        # Log metrics using logger.log_metrics method
        metrics = {
            "val/accuracy": accuracy,
            "val/precision": precision,
            "val/recall": recall,
            "val/f1_score": f1_score,
        }
        self.logger.log_metrics(metrics, step=self.global_step)

        # Log confusion matrix (as an image)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(conf_mat)
        plt.colorbar(cax)
        
        self.logger.experiment.add_image('Confusion Matrix', fig, global_step=self.current_epoch)

        self.val_step_outputs.clear()

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
tb_logger = TensorBoardLogger("lightning_logs", name="cricket_experiment")
# comet_logger = CometLogger(api_key=os.environ.get("COMET_API_KEY"),save_dir="lightning_logs_comet", project_name="cricket_experiment")
# Initialize the Trainer
trainer = Trainer(
    max_epochs=100,
    devices=2, 
    accelerator="gpu",
    strategy="ddp",
    logger=tb_logger,
    gradient_clip_val=1.0
)
num_workers = 40  # or another value based on your system's specifications
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=num_workers)

# Train the model
trainer.fit(classifier, train_loader, test_loader)

# Save the best model
model_save_path = f"{tb_logger.log_dir}/best_finetuned_ast_cricket_data.pt"
torch.save(classifier.model.state_dict(), model_save_path)
