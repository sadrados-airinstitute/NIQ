import torch
import torch.optim as optim
from torch import nn

class ModelTrainer:
    def __init__(self, model, dataloader, optimizer, criterion):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion

    def train_model(self, model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4):
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()  # Use BCEWithLogitsLoss for multi-label classification
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, text, labels in train_loader:
                images, text, labels = images.to(device), text.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(images, text)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

            # Validate after every epoch
            validate_model(model, val_loader)

            # Optionally, save the model after every epoch
            save_model(model, epoch)

    def validate_model(self, model, val_loader):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, text, labels in val_loader:
                images, text, labels = images.to(device), text.to(device), labels.to(device)

                # Forward pass
                outputs = model(images, text)

                # Calculate loss
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss/len(val_loader)}")

    def save_model(self, model, epoch, path="model_checkpoint.pth"):
        torch.save(model.state_dict(), path)
        print(f"Model saved at epoch {epoch}.")
