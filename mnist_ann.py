import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Hyperparameters
learning_rate = 1e-4
batch_size = 32
epochs = 10 

# Define the ANN Model
class MNIST_ANN(nn.Module):
    def __init__(self):
        super(MNIST_ANN, self).__init__()
        # Input image is 28x28 = 784 pixels
        # 3 hidden layers with 64, 64, and 32 units respectively
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10) # Output layer for 10 classes (digits 0-9)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits

def main():
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize pixel values to [-1, 1]
    ])

    # Load Full MNIST Training Dataset (which has 60,000 samples)
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # The requirement asks for 50k train. So we randomly split the 60k into 50k and discard/keep 10k.
    train_dataset, _ = random_split(full_train_dataset, [50000, 10000])
    
    # Load MNIST Test Dataset (which has exactly 10,000 samples)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNIST_ANN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Model Architecture:\n{model}")
    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")
    print(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}")
    
    # Optional: Basic Training Loop
    """
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
    """

if __name__ == "__main__":
    main()
