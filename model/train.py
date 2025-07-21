import torch
import torch.optim as optim
from torch import nn
from torch.optim import AdamW
from transformers import BertTokenizer, BertForTokenClassification
from tqdm import tqdm
import os
from typing import Optional
import pandas as pd
from model.entity_recognition_model import EntityRecognitionModel
from utils.create_dataset import EntityRecognitionDataset, ClassifierDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model.evaluation import EntityRecognitionModelEvaluator, ClassifierModelEvaluator
from utils.utils import save_checkpoint_entity_recognition, save_checkpoint_classifier
class EntityRecognitionModelTrainer:
    
    def __init__(self, csv_path: str, model_path: Optional[str] = None, model_name: Optional[str] = "bert-base-cased", epochs: Optional[int] = 10, learning_rate: Optional[float] = 5e-5):
        """
        Initializes the Entity Recognition Model Trainer for fine-tuning.

        Args:
            csv_path (str): Path to CSV file with training data.
            model_path (Optional[str]): Directory to save or load the model from.
            model_name (Optional[str]): Hugging Face model identifier (default: "bert-base-cased").
            epochs (Optional[int]): Number of training epochs (default: 10).
            learning_rate (Optional[float]): Learning rate (default: 5e-5).
        """
        
         # Load model/tokenizer
        if model_path and os.path.exists(model_path):
            self.model = BertForTokenClassification.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        elif model_name:
            self.model = BertForTokenClassification.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            raise ValueError("You must provide either a valid model_path or a model_name.")
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
       
        self.dataset = EntityRecognitionDataset(csv_path=csv_path)
        
        # Ensure label2id/id2label in model config matches the dataset
        self.model.config.label2id = self.dataset.label2id
        self.model.config.id2label = {i: l for l, i in self.dataset.label2id.items()}
        self.num_labels = len(self.dataset.label2id)
        
        # Split examples into train and validation
        train_ex, val_ex = train_test_split(self.dataset.examples, test_size=0.2, random_state=42)

        # Create datasets
        self.train_dataset = EntityRecognitionDataset(examples=train_ex, label2id=label2id)
        self.val_dataset = EntityRecognitionDataset(examples=val_ex, label2id=label2id)

        # Create dataloader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Store hyperparameters
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Initialize ModelEvaluator
        self.evaluator = EntityRecognitionModelEvaluator(
            model=self.model,
            label_map=self.model.config.id2label,
            evaluation_data_loader=self.val_dataloader,
            tokenizer=self.tokenizer
        )
        
       

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

            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
            avg_loss = running_loss / len(self.train_dataloader)
            accuracy = correct_predictions / total_samples
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            # Validation after every epoch
            self.validate_model()

            # Optionally, save the model after every epoch
            save_checkpoint_entity_recognition(model=self.model, tokenizer=self.tokenizer, label2id=self.dataset.label2id, optimizer=self.optimizer, epoch=epoch, loss=avg_loss)

    def validate_model(self):
        """
        Validates the model on the validation set using token-level and entity-level metrics.
        """
        metrics = self.evaluator.evaluate(self.device)
    


class ClassifierModelTrainer:
    
    def __init__(self, csv_path: str, model_path: Optional[str] = None, model_name: Optional[str] = "bert-base-cased", epochs: Optional[int] = 10, learning_rate: Optional[float] = 5e-5):
        """
        Initializes the Classifier Model Trainer for fine-tuning.

        Args:
            csv_path (str): Path to CSV file with training data.
            model_path (Optional[str]): Directory to save or load the model from.
            model_name (Optional[str]): Hugging Face model identifier (default: "bert-base-cased").
            epochs (Optional[int]): Number of training epochs (default: 10).
            learning_rate (Optional[float]): Learning rate (default: 5e-5).
        """
        
        # Load model/tokenizer
        if model_path and os.path.exists(model_path):
            self.model = BertForTokenClassification.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        elif model_name:
            self.model = BertForTokenClassification.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            raise ValueError("You must provide either a valid model_path or a model_name.")
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
       
        self.dataset = ClassifierDataset(csv_path=csv_path)
        
        # Split into train/validation
        train_ex, val_ex = train_test_split(dataset.examples, test_size=0.2, stratify=[ex['label'] for ex in dataset.examples], random_state=42)
        self.train_dataset = ClassifierDataset(examples=train_ex, tokenizer=self.tokenizer)
        self.val_dataset = ClassifierDataset(examples=val_ex, tokenizer=self.tokenizer)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.epochs = epochs
        self.batch_size = batch_size
        self.evaluator = ClassifierModelEvaluator(model=self.model, evaluation_data_loader=self.val_dataloader, tokenizer=self.tokenizer)

    def train_model(self):
        """
        Trains the binary classifier.
        """
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

            avg_loss = running_loss / len(self.train_dataloader)
            accuracy = correct_predictions / total_samples
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            # Validation
            self.validate_model()

            # Save
            save_checkpoint_classifier(
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=avg_loss,
                output_dir=f"checkpoints/classifier_epoch_{epoch+1}"
            )

    def validate_model(self):
        """
        Evaluates the classifier using standard binary classification metrics.
        """
        metrics = self.evaluator.evaluate(self.device)
        print("Validation metrics:", metrics)