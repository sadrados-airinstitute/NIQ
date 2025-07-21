import torch
import numpy as np
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional


class EntityRecognitionModelEvaluator:
    def __init__(self, model, label_map: dict, evaluation_data_loader, tokenizer):
        """
        Initializes the evaluator.

        Args:
            model: Trained NER model (e.g., BertForTokenClassification).
            label_map (dict): Mapping from label IDs to label strings (id2label).
            evaluation_data_loader: DataLoader for evaluation dataset.
            tokenizer: Tokenizer used during training.
        """
        self.model = model
        self.id2label = label_map
        self.tokenizer = tokenizer
        self.dataloader = evaluation_data_loader

    def evaluate(self, device):
        """
        Evaluates the NER model using both entity-level and token-level metrics.

        Args:
            device: torch.device object (CPU or CUDA).

        Returns:
            dict: Evaluation metrics.
        """
        self.model.eval()
        self.model.to(device)

        true_entities: List[List[str]] = []
        pred_entities: List[List[str]] = []

        true_tokens_flat = []
        pred_tokens_flat = []

        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)

                for i in range(len(labels)):
                    word_ids = self.tokenizer.word_ids(batch_index=i)

                    true_seq = []
                    pred_seq = []

                    for j, word_id in enumerate(word_ids):
                        if word_id is None:
                            continue
                        if labels[i][j].item() == -100:
                            continue

                        true_id = labels[i][j].item()
                        pred_id = predictions[i][j].item()

                        true_label = self.id2label[true_id]
                        pred_label = self.id2label[pred_id]

                        # For entity-level metrics
                        true_seq.append(true_label)
                        pred_seq.append(pred_label)

                        # For token-level metrics
                        true_tokens_flat.append(true_id)
                        pred_tokens_flat.append(pred_id)

                    true_entities.append(true_seq)
                    pred_entities.append(pred_seq)

        # Entity-level (seqeval)
        entity_precision = precision_score(true_entities, pred_entities)
        entity_recall = recall_score(true_entities, pred_entities)
        entity_f1 = f1_score(true_entities, pred_entities)
        entity_report = classification_report(true_entities, pred_entities)

        print("Entity-level classification report (BIO spans):")
        print(entity_report)
        
        # Token-level (sklearn)
        token_accuracy = accuracy_score(true_tokens_flat, pred_tokens_flat)
        token_precision, token_recall, token_f1, _ = precision_recall_fscore_support(
            true_tokens_flat, pred_tokens_flat, average="weighted", zero_division=1
        )

        # Confusion Matrix
        self.plot_confusion_matrix(
            y_true=true_tokens_flat,
            y_pred=pred_tokens_flat,
            labels=[self.id2label[i] for i in sorted(self.id2label.keys())]
        )

        return {
            "entity_precision": entity_precision,
            "entity_recall": entity_recall,
            "entity_f1": entity_f1,
            "token_accuracy": token_accuracy,
            "token_precision": token_precision,
            "token_recall": token_recall,
            "token_f1": token_f1
        }

    def plot_confusion_matrix(self, y_true, y_pred, labels: Optional[List[str]] = None):
        """
        Plots a token-level confusion matrix.

        Args:
            y_true (List[int]): True label IDs.
            y_pred (List[int]): Predicted label IDs.
            labels (List[str]): Label names ordered by ID.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Token-Level Confusion Matrix")
        plt.tight_layout()
        plt.show()
        
class ClassifierModelEvaluator:
    """
    Evaluator for a binary classification model (e.g., relation classifier for triplets).
    """

    def __init__(self, model, evaluation_data_loader, tokenizer=None):
        """
        Args:
            model: Trained Hugging Face model (e.g., BertForSequenceClassification).
            evaluation_data_loader: PyTorch DataLoader for evaluation set.
            tokenizer: Optional tokenizer (not used here, included for consistency).
        """
        self.model = model
        self.dataloader = evaluation_data_loader
        self.tokenizer = tokenizer

    def evaluate(self, device) -> dict:
        """
        Evaluates the classifier using standard binary metrics.

        Args:
            device (torch.device): 'cuda' or 'cpu'

        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        self.model.to(device)

        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)

                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predictions.cpu().tolist())

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)
        report = classification_report(y_true, y_pred, digits=4)

        print("Binary classification report:\n")
        print(report)

        self.plot_confusion_matrix(y_true, y_pred, labels=["Not Linked (0)", "Linked (1)"])

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], labels: Optional[List[str]] = None):
        """
        Plots a confusion matrix for binary classification.

        Args:
            y_true (List[int]): Ground truth labels.
            y_pred (List[int]): Predicted labels.
            labels (List[str], optional): Class label names.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()