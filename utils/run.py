import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Define a CNN architecture suitable for CIFAR-10
class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarCNN, self).__init__()
        # Input: 3x32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Changed input channels to 3
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32x16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x8x8
        self.flatten = nn.Flatten()
        # Adjusted FC layer input size based on conv/pool output (64 channels * 8 height * 8 width)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# Function to train the model (remains the same)
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):  # Added device
    model.train()
    model.to(device)  # Move model to device
    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move data to device

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {running_loss / len(train_loader):.4f}')


# Function to test the model (remains mostly the same)
def test_model(model, test_loader, device='cpu'):  # Added device
    model.eval()
    model.to(device)  # Move model to device
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the test images: {100 * correct / total:.2f} %')


# Main function to run the training and testing
# Changed default data_dir to './data' for CIFAR-10 structure
def run(data_dir='../data', num_epochs=10, batch_size=64, lr=0.001):
    # Check for available device (MPS for Mac, CUDA for others, fallback CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define transforms for CIFAR-10, including normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
    ])

    # CIFAR-10 dataset
    # Changed to datasets.CIFAR10, updated root path, kept download=True
    try:
        train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
    except Exception as e:
        print(f"Error loading CIFAR-10 dataset from {data_dir}: {e}")
        print("Please ensure the directory exists and you have internet connectivity for download.")
        return

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2)  # Added num_workers
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2)  # Added num_workers

    # Initialize the model (using the new CifarCNN)
    model = CifarCNN()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs, device=device)

    # Test the model
    test_model(model, test_loader, device=device)


if __name__ == '__main__':
    # Create data directory if it doesn't exist
    if not os.path.exists('../data'):
        os.makedirs('./data')
    run()