import os
import random
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(image_dir, text_file, test_size=0.2, val_size=0.1, seed=42):
    """
    Split data into training, validation, and test sets.
    Args:
        image_dir (str): Directory containing the images.
        text_file (str): Path to the CSV file with image file paths and labels (e.g., nutritional information).
        test_size (float): Proportion of the data to include in the test set.
        val_size (float): Proportion of the training data to include in the validation set.
        seed (int): Random seed for reproducibility.
    
    Returns:
        train_data, val_data, test_data: DataFrames containing paths to images and their corresponding labels.
    """
    # Load image paths and corresponding labels (assumes CSV with 'image_path' and 'label' columns)
    data = pd.read_csv(text_file)
    
    # Split into train+val and test sets
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=seed)
    
    # Split train+val into train and validation sets
    train_data, val_data = train_test_split(train_val_data, test_size=val_size, random_state=seed)
    
    # Optionally, create directories to organize the images for each set
    create_image_subdirs(image_dir, train_data, val_data, test_data)
    
    return train_data, val_data, test_data

def create_image_subdirs(image_dir, train_data, val_data, test_data):
    """
    Create subdirectories for each dataset (train, val, test) and move the images accordingly.
    """
    # Create directories if they don't exist
    os.makedirs(os.path.join(image_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(image_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(image_dir, 'test'), exist_ok=True)
    
    # Move images to the respective directories
    for index, row in train_data.iterrows():
        shutil.copy(os.path.join(image_dir, row['image_path']), os.path.join(image_dir, 'train', row['image_path']))
    for index, row in val_data.iterrows():
        shutil.copy(os.path.join(image_dir, row['image_path']), os.path.join(image_dir, 'val', row['image_path']))
    for index, row in test_data.iterrows():
        shutil.copy(os.path.join(image_dir, row['image_path']), os.path.join(image_dir, 'test', row['image_path']))

def get_data_paths(dataframe):
    """
    Extract file paths and labels from a DataFrame to be used in a DataLoader.
    """
    image_paths = dataframe['image_path'].tolist()
    labels = dataframe['label'].tolist()
    return image_paths, labels