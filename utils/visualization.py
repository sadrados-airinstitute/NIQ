import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_training_loss(training_loss, validation_loss):
    """
    Plot training and validation loss over epochs.
    
    Args:
        training_loss (list): List of training losses.
        validation_loss (list): List of validation losses.
    """
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot a confusion matrix to visualize performance.
    
    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        labels (list): Class labels for confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_precision_recall_curve(precision, recall, thresholds):
    """
    Plot Precision-Recall curve.
    
    Args:
        precision (list): List of precision values.
        recall (list): List of recall values.
        thresholds (list): List of thresholds.
    """
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.show()