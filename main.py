import json, argparse
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from dataset import CustomDataset
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pprint
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class CricketClassifier(LightningModule):
    def __init__(self, label2id, id2label, num_classes, train_loader_len, test_loader_len):
        super().__init__()
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593", label2id=label2id, id2label=id2label ,num_labels=num_classes,ignore_mismatched_sizes=True)
        self.criterion = nn.CrossEntropyLoss()
        self.training_step_outputs = []
        self.val_step_outputs = []
        self.train_dataloader_len = train_loader_len
        self.test_dataloader_len = test_loader_len

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
        _, predicted = torch.max(logits, dim=1)
        # Log and print gradients
        # self.log_gradients()
        metrics = {"train_loss_step": loss.item()}
        self.logger.log_metrics(metrics, step=self.global_step)
        train_outputs = {"loss": loss, "labels": labels, "predictions": predicted}
        self.training_step_outputs.append(train_outputs)
        # print(f"Train Loss (step): {loss:.4f}", end="\r")
        return train_outputs

    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics = {"train_loss_epoch": avg_loss.item()}
        self.logger.log_metrics(metrics, step=self.current_epoch)
        
        all_labels = torch.cat([x['labels'] for x in outputs], dim=0)
        all_predictions = torch.cat([x['predictions'] for x in outputs], dim=0)

        # Calculate accuracy
        accuracy = (all_predictions == all_labels).sum().item() / all_labels.size(0)

        # Calculate precision, recall, F1-score, and confusion matrix
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels.cpu(), all_predictions.cpu(), average='macro')

        # Log metrics using logger.log_metrics method
        metrics = {
            "train/accuracy": accuracy,
            "train/precision": precision,
            "train/recall": recall,
            "train/f1_score": f1_score,
        }
        self.logger.log_metrics(metrics, step=self.global_step)

        # # Log confusion matrix (as an image)
        # cf_matrix = confusion_matrix(all_labels.cpu().numpy(), all_predictions.cpu().numpy())
        # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None])
        # fig, ax = plt.subplots(figsize=(12, 7))
        # sn.heatmap(df_cm, annot=True, ax=ax)

        # # Convert the figure to a numpy array
        # canvas = FigureCanvas(fig)
        # canvas.draw()
        # img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        # img_array = img_array.reshape(canvas.get_width_height()[::-1] + (3,))

        # # Close the figure to free resources
        # plt.close(fig)
        # self.logger.experiment.add_image('Confusion Matrix-Train', img_array , global_step=self.current_epoch)
        
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

        # Log metrics using logger.log_metrics method
        metrics = {
            "val/accuracy": accuracy,
            "val/precision": precision,
            "val/recall": recall,
            "val/f1_score": f1_score,
        }
        self.logger.log_metrics(metrics, step=self.global_step)

        # # Log confusion matrix (as an image)
        # cf_matrix = confusion_matrix(all_labels.cpu().numpy(), all_predictions.cpu().numpy())
        # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None])
        # fig, ax = plt.subplots(figsize=(12, 7))
        # sn.heatmap(df_cm, annot=True, ax=ax)

        # # Convert the figure to a numpy array
        # canvas = FigureCanvas(fig)
        # canvas.draw()
        # img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        # img_array = img_array.reshape(canvas.get_width_height()[::-1] + (3,))

        # # Close the figure to free resources
        # plt.close(fig) 
        # self.logger.experiment.add_image('Confusion Matrix-Val', img_array , global_step=self.current_epoch)

        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-5)
        num_training_steps = len(self.train_dataloader_len) * self.trainer.max_epochs
        warmup_steps = int(num_training_steps * 0.1)
        scheduler = {
            'scheduler': OneCycleLR(optimizer, max_lr=3e-5, total_steps=num_training_steps, anneal_strategy='linear', pct_start=warmup_steps/num_training_steps, div_factor=25.0, final_div_factor=10000.0),
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

# Rest of the code remains the same


def main(train_data_path, test_data_path, epochs):
    # Load the dataset
    train_set = CustomDataset(data_path=train_data_path, type='genus')

    label2id = train_set.label2id
    id2label = train_set.id2label
    pprint.pprint(label2id)
    num_classes = len(label2id)

    test_set = CustomDataset(data_path=test_data_path, label2id=label2id, id2label=id2label, type='genus')

    lr_monitor = LearningRateMonitor(logging_interval="step")
    # Set up TensorBoard logger
    tb_logger = TensorBoardLogger("lightning_logs", name="cricket_experiment")
    tb_logger.log_hyperparams({"id2label" : json.dumps(id2label)})
    # comet_logger = CometLogger(api_key=os.environ.get("COMET_API_KEY"),save_dir="lightning_logs_comet", project_name="cricket_experiment")
    # Initialize the Trainer
    trainer = Trainer(
        max_epochs=epochs,
        devices=2, 
        accelerator="gpu",
        strategy="ddp",
        logger=tb_logger,
        gradient_clip_val=4.0,
        accumulate_grad_batches=4,
        callbacks=[lr_monitor]
    )
    num_workers = 40  # or another value based on your system's specifications
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=num_workers)
        # Initialize the LightningModule
    classifier = CricketClassifier(label2id, id2label, num_classes, len(train_loader), len(test_loader))
    # Train the model
    trainer.fit(classifier, train_loader, test_loader)

    # Save the best model
    model_save_path = f"{tb_logger.log_dir}/best_finetuned_ast_cricket_data.pt"
    torch.save(classifier.model.state_dict(), model_save_path)


if __name__ == '__main__':
    # Load the configuration file
    with open('config.json') as config_file:
        config = json.load(config_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default=config['main']['train_data_path'] ,help='Path to the train data directory')
    parser.add_argument('--test_data_path', default=config['main']['test_data_path'] ,help='Path to the test data directory')
    parser.add_argument('--epochs', default=config['main']['epochs'] ,help='Number of epochs to train the model')
    args = parser.parse_args()

    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    epochs = args.epochs

    main(train_data_path, test_data_path, int(epochs))