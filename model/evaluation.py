from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ModelEvaluator:
    def __init__(self, model, evaluation_data_loader):
        self.model = model
        self.evaluation_data_loader = evaluation_data_loader
        
    def evaluate(self, model, dataloader, device):
        """
        Evaluates the model on the given dataset (validation or test).
        Calculates accuracy, precision, recall, F1 score, and plots a confusion matrix.

        Args:
            model: The trained model to evaluate.
            dataloader: A PyTorch DataLoader providing the evaluation data.
            device: The device to run the model on (CPU or GPU).
        
        Returns:
            metrics (dict): A dictionary containing evaluation metrics.
        """
        model.eval()  # Set the model to evaluation mode
        y_true = []
        y_pred = []

        with torch.no_grad():  # No need to calculate gradients during evaluation
            for images, texts, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                # Assuming model output is a tensor of predicted labels
                outputs = model(images, texts)  # Replace with actual model inference code
                predicted = torch.argmax(outputs, dim=1)  # Get the predicted labels
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
        
        # Classification report
        report = classification_report(y_true, y_pred)
        print("Classification Report:\n", report)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm)

        # Return metrics as a dictionary
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        return metrics

    def plot_confusion_matrix(self, cm, labels=None):
        """
        Plots the confusion matrix using seaborn.

        Args:
            cm (numpy array): Confusion matrix to plot.
            labels (list): List of class labels (optional).
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        plt.show()