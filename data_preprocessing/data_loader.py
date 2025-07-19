import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

class MultimodalDataset(Dataset):
    def __init__(self, image_paths, text_data, labels, transform=None):
        self.image_paths = image_paths  # List of image paths
        self.text_data = text_data  # Corresponding text data (OCR results or structured text)
        self.labels = labels  # Nutritional information (e.g., calories, sugar, etc.)
        self.transform = transform  # Transformations to apply to the image data

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx])
        
        # Load the corresponding text (OCR or structured data)
        text = self.text_data[idx]
        
        # Load the label (nutritional info)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, text, label

def get_data_loaders(image_dir, text_data, labels, batch_size=32, val_split=0.2):
    # Split data into training and validation sets (for example)
    total_size = len(image_dir)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    train_image_paths = image_dir[:train_size]
    val_image_paths = image_dir[train_size:]
    
    train_text_data = text_data[:train_size]
    val_text_data = text_data[train_size:]
    
    train_labels = labels[:train_size]
    val_labels = labels[train_size:]
    
    # Define image transformations (e.g., resize, normalization, augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
    ])
    
    # Create the training and validation datasets
    train_dataset = MultimodalDataset(train_image_paths, train_text_data, train_labels, transform)
    val_dataset = MultimodalDataset(val_image_paths, val_text_data, val_labels, transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader
