import torch
import json
from transformers import BertForTokenClassification, BertTokenizer
import os

def load_from_checkpoint_entity_recognition(checkpoint_path: str):
    """
    Loads a model, tokenizer, and label map from a checkpoint directory.

    Args:
        checkpoint_path (str): path to saved model directory

    Returns:
        model, tokenizer, label2id
    """
    model = BertForTokenClassification.from_pretrained(checkpoint_path)
    tokenizer = BertTokenizer.from_pretrained(checkpoint_path)

    with open(os.path.join(checkpoint_path, "label2id.json"), "r") as f:
        label2id = json.load(f)
        model.config.label2id = label2id
        model.config.id2label = {v: k for k, v in label2id.items()}

    return model, tokenizer, label2id

def save_checkpoint_entity_recognition(model, tokenizer, label2id, optimizer, epoch, loss, output_dir="models/ner_model"):
        """
        Saves model, tokenizer, label map, and optimizer state.

        Args:
            model: Hugging Face model
            tokenizer: Hugging Face tokenizer
            label2id: label-to-id dictionary
            optimizer: PyTorch optimizer
            epoch: current epoch
            loss: validation or training loss
            output_dir: directory to store the checkpoint
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save label2id and id2label
        with open(os.path.join(output_dir, "label2id.json"), "w") as f:
            json.dump(label2id, f)

        # Save optimizer and training state (optional)
        torch.save({
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss
        }, os.path.join(output_dir, "training_state.pt"))

        print(f"Model checkpoint saved at: {output_dir}")
        
        
def save_checkpoint_classifier(model, tokenizer, optimizer, epoch, loss, output_dir="checkpoints/classifier"):
    """
    Saves the classifier model, tokenizer, and optimizer state for resuming training or inference.

    Args:
        model (PreTrainedModel): Hugging Face classification model (e.g., BertForSequenceClassification).
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
        optimizer (torch.optim.Optimizer): Optimizer used in training.
        epoch (int): Current training epoch.
        loss (float): Loss at current epoch (optional for tracking).
        output_dir (str): Directory to save the checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training state (e.g., optimizer, epoch, loss)
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "optimizer_state_dict": optimizer.state_dict()
    }, os.path.join(output_dir, "training_state.pt"))

    print(f"Classifier checkpoint saved at: {output_dir}")
    
def load_from_checkpoint_classifier(checkpoint_dir: str, load_optimizer: bool = False):
    """
    Loads a classifier model, tokenizer, and optionally optimizer state from a saved checkpoint.

    Args:
        checkpoint_dir (str): Path to the directory containing the checkpoint.
        load_optimizer (bool): Whether to load the optimizer state and training metadata.

    Returns:
        model (BertForSequenceClassification)
        tokenizer (BertTokenizer)
        optimizer_state (dict) or None: optimizer state and training metadata (epoch, loss) if requested
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    # Load model and tokenizer
    model = BertForSequenceClassification.from_pretrained(checkpoint_dir)
    tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)

    # Optionally load optimizer state
    optimizer_state = None
    training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if load_optimizer:
        if not os.path.exists(training_state_path):
            raise FileNotFoundError(f"Missing training_state.pt in {checkpoint_dir}")
        optimizer_state = torch.load(training_state_path)

    return model, tokenizer, optimizer_state

def save_config(config, config_file="models/config.json"):
    """
    Save the model configuration (hyperparameters, settings, etc.) to a JSON file.
    
    Args:
        config (dict): The configuration dictionary.
        config_file (str): The path to save the configuration file.
    """
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved at {config_file}")

def load_config(config_file="models/config.json"):
    """
    Load the configuration dictionary from a JSON file if it exists.
    
    Args:
        config_file (str): The path to the configuration file.
    
    Returns:
        dict: The loaded configuration dictionary if the file exists.
        None: If the config file does not exist.
    """
    # Check if the configuration file exists
    if not os.path.exists(config_file):
        print(f"Configuration file '{config_file}' does not exist.")
        return None
    
    # Load the configuration from the file
    with open(config_file, 'r') as f:
        config = json.load(f)

    print(f"Configuration loaded from {config_file}")
    return config