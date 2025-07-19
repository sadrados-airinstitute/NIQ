import torch
import json

def save_model(model, model_name="final_model.pth"):
    """
    Save the model's state dict (weights) to the models directory.
    
    Args:
        model (torch.nn.Module): The model to be saved.
        model_name (str): The name of the model file to save.
    """
    # Define the path for saving the model
    model_path = f"models/{model_name}"
    
    # Save only the state_dict (recommended in PyTorch)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

def load_model(model, model_name="final_model.pth"):
    """
    Load the model's state dict (weights) from the models directory.
    
    Args:
        model (torch.nn.Module): The model to load the state dict into.
        model_name (str): The name of the model file to load.
    
    Returns:
        model (torch.nn.Module): The model with loaded state dict.
    """
    # Define the path to the saved model
    model_path = f"models/{model_name}"
    
    # Load the model's state dict
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    
    return model

def save_checkpoint(model, optimizer, epoch, loss, filename="models/checkpoint.pth"):
    """
    Save the model checkpoint including the model state, optimizer state, epoch, and loss.
    
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): The current epoch number.
        loss (float): The current loss value.
        filename (str): The file path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")

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

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility across experiments.
    
    Args:
        seed (int): The seed value to use for randomness.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)