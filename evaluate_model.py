import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np

# Import the model architecture from our previous file
from mnist_ann import MNIST_ANN

def main():
    # Hyperparameters
    learning_rate = 1e-4
    batch_size = 32
    epochs = 10 
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load and split dataset
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_dataset, _ = random_split(full_train_dataset, [50000, 10000])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize device, model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNIST_ANN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting Training...")
    # 1. Train the model and record loss per epoch
    train_losses = []
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
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Plot Loss vs Epoch
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, marker='o', color='b', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_vs_epoch.png')
    plt.close()
    print("\nSaved: loss_vs_epoch.png")

    print("\nEvaluating the Model...")
    # 2. Evaluate and compute metrics
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Calculate Precision, Recall, F1 Score, and Accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    
    print("\n--- Model Evaluation Data (Test Set) ---")
    print(f"Accuracy:  {acc:.4f} ({(acc*100):.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Saved: confusion_matrix.png")

    # 3. Show features layer-wise
    # Feed a single sample image into the model to extract and visualize immediate activations
    sample_image, sample_label = test_dataset[0]
    sample_image_input = sample_image.unsqueeze(0).to(device) # Expand dims to batch size 1
    
    activations = []
    x = model.flatten(sample_image_input)
    
    # Store the output after each ReLU activation layer to represent "features"
    for layer in model.network:
        x = layer(x)
        if hasattr(layer, 'out_features') or isinstance(layer, nn.ReLU):
            # Only record activations outputted directly after activation functions
            if isinstance(layer, nn.ReLU):
                activations.append(x.cpu().detach().numpy().squeeze())
            
    # Visualize intermediate layer activations as heatmaps
    num_layers = len(activations)
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 2 * num_layers))
    
    if num_layers == 1:
        axes = [axes]
    
    for i, act in enumerate(activations):
        # We reshape array into (1, num_units) for barcode-style heatmap visualization
        act_vis = act.reshape(1, -1)
        sns.heatmap(act_vis, ax=axes[i], cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
        axes[i].set_title(f'Hidden Layer {i+1} Activations ({act.shape[0]} feature units)')

    plt.tight_layout()
    plt.savefig('layer_wise_features.png')
    plt.close()
    print("Saved: layer_wise_features.png")

if __name__ == "__main__":
    main()
