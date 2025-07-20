import torch
import torch.optim as optim
from torch import nn
from torch.optim import AdamW
from transformers import BertTokenizer, BertForTokenClassification
from tqdm import tqdm
import os

class EntityRecognitionModelTrainer:
    
    def __init__(self, model_path=None, model_name="dbmdz/bert-large-cased-finetuned-conll03-english", 
                 dataloader=None, device=None, criterion=None, optimizer=None):
        """
        Initializes the Entity Recognition Model Trainer for fine-tuning.

        Args:
            model_path (str): Path to the directory where the trained model is saved locally (optional).
            model_name (str): The name of the pre-trained model from Hugging Face if model_path is not provided.
            dataloader: The DataLoader that provides the data in batches.
            device: The device (CPU or GPU) where the model should run.
            criterion: Loss function for training (default: CrossEntropyLoss).
            optimizer: Optimizer for training (default: AdamW).
        """
        if model_path and os.path.exists(model_path):
            # Load model and tokenizer from a local folder
            self.model = BertForTokenClassification.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        else:
            # Load pre-trained model and tokenizer from Hugging Face model hub
            self.model = BertForTokenClassification.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Initialize other properties
        self.dataloader = dataloader
        self.device = device
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()  # Default to CrossEntropyLoss for NER tasks
        self.optimizer = optimizer if optimizer else AdamW(self.model.parameters(), lr=5e-5)  # Default AdamW optimizer


    def train_model(self, num_epochs=10):
        """
        Fine-tunes the NER model on the provided dataset.

        Args:
            num_epochs: Number of epochs to train the model.
        """

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            total_samples = 0
            correct_predictions = 0

            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Calculate running loss
                running_loss += loss.item()
                total_samples += input_ids.size(0)

                # Calculate correct predictions (for evaluation)
                preds = torch.argmax(logits, dim=-1)
                correct_predictions += (preds == labels).sum().item()

            # Print the average loss for the epoch
            avg_loss = running_loss / len(self.dataloader)
            accuracy = correct_predictions / total_samples
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            # Validation after every epoch
            self.validate_model()

            # Optionally, save the model after every epoch
            self.save_model(epoch)

    def validate_model(self):
        """
        Validates the model on the validation set.
        """
        self.model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item()
                total_samples += input_ids.size(0)

                # Calculate correct predictions
                preds = torch.argmax(logits, dim=-1)
                correct_predictions += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(self.dataloader)
        val_accuracy = correct_predictions / total_samples
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    def save_model(self, epoch, path="data/model_checkpoint.pth"):
        """
        Saves the trained model to the specified path.
        
        Args:
            epoch: The epoch at which the model is saved.
            path: The path to save the model.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at epoch {epoch}.")