from torchvision import transforms

def get_image_transforms(is_train=True):
    """
    Get the image transformations for both training and validation.
    For training, we include augmentations to help the model generalize better.
    For validation, we typically don't use augmentations.

    Args:
        is_train (bool): Whether to get transformations for training or validation.
    
    Returns:
        transform (torchvision.transforms.Compose): A sequence of transformations.
    """
    if is_train:
        # Define the transformations for training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224 (typical for CNNs like ResNet)
            transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
            transforms.RandomRotation(20),  # Random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (e.g., for pre-trained models)
        ])
    else:
        # Define the transformations for validation/test (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
        ])
    
    return transform
