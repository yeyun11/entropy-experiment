import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from entcal import calculate_entropy
from skimage.restoration import non_local_means, estimate_sigma


def get_cnn(hidden_channels=[16, 16, "A2", 32, 32, "A2", 64, 64, "A2", 128, 128, "AA"]):
    layers = []
    prev_dim = 3

    for h in hidden_channels:
        if isinstance(h, str):
            if h == "A2":
                layers.append(nn.AvgPool2d(2))
            elif h == "AA":
                layers.append(nn.AdaptiveAvgPool2d(1))
        else:
            layers.append(nn.Conv2d(prev_dim, h, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(h))
            layers.append(nn.ReLU())
            prev_dim = h
    layers.append(nn.Flatten())
    layers.append(nn.Linear(prev_dim, 1))
    
    return nn.Sequential(*layers)


def train_epoch(model, train_loader, criterion, optimizer, device, num_bins, use_noise):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for data, _ in train_loader:
        if use_noise:
            noise_data = []
            for image in data:
                image = image.numpy()
                sigma = estimate_sigma(image, average_sigmas=True)
                denoised = non_local_means.denoise_nl_means(image, sigma=sigma, h=0.025, channel_axis=0)
                noise = image - denoised
            noise_data.append(torch.from_numpy(noise))
            data = torch.stack(noise_data)
        data = data.to(device)
        targets = calculate_entropy(data, num_bins=num_bins, adaptive=True)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, test_loader, criterion, device, num_bins, use_noise):
    """
    Evaluate the model on test data.
    
    Args:
        model: The neural network model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        tuple: (average loss, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            if use_noise:
                noise_data = []
                for image in data:
                    image = image.numpy()
                    sigma = estimate_sigma(image, average_sigmas=True)
                    denoised = non_local_means.denoise_nl_means(image, sigma=sigma, h=0.025, channel_axis=0)
                    noise = image - denoised
                noise_data.append(torch.from_numpy(noise))
                data = torch.stack(noise_data)
            data = data.to(device)
            targets = calculate_entropy(data, num_bins=num_bins, adaptive=True)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return total_loss / num_batches, np.array(all_predictions), np.array(all_targets)


def plot_results(predictions, targets, save_path=None):
    """
    Plot the results of entropy prediction.
    
    Args:
        predictions: Model predictions
        targets: True entropy values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Scatter plot of predictions vs targets
    plt.subplot(1, 3, 1)
    plt.scatter(targets, predictions, alpha=0.6)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel('True Entropy')
    plt.ylabel('Predicted Entropy')
    plt.title('Predictions vs True Values')
    plt.grid(True, alpha=0.3)
    
    # Histogram of residuals
    plt.subplot(1, 3, 2)
    residuals = predictions - targets
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals (Predicted - True)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    # Learning curve (if available)
    plt.subplot(1, 3, 3)
    plt.plot(targets, label='True', alpha=0.7)
    plt.plot(predictions, label='Predicted', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Entropy')
    plt.title('Entropy Values Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.close()
    #plt.show()


def main(crop_size, num_bins, batch_size, num_epochs, learning_rate, cnn_hidden_channels, seed, use_noise):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create train and test dataloaders
    print("Creating dataloaders...")
    train_dataset = datasets.CIFAR100(
        root='./cifar-100',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
        ])
    )
    test_dataset = datasets.CIFAR100(
        root='./cifar-100',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = get_cnn(cnn_hidden_channels)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Training loop
    print("Starting training...")
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, num_bins, use_noise)
        
        # Evaluate
        test_loss, predictions, targets = evaluate(model, test_loader, criterion, device, num_bins, use_noise)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Log metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {train_loss:.6f}, "
                f"Test Loss: {test_loss:.6f}")
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'train_loss': train_loss,
            }, 'best_entropy_model.pth')
    
    # Final evaluation
    print("\nFinal Evaluation:")
    print(f"Best test loss: {best_test_loss:.6f}")
    
    # Calculate additional metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    
    # Plot results
    tag_use_noise = "noise" if use_noise else "image"
    plot_results(predictions, targets, 'entropy_prediction_results-cifar100-{}.png'.format(tag_use_noise))
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_curves-cifar100-{tag_use_noise}.png', dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()
    
    print("Training completed!")


if __name__ == "__main__":

    # Training parameters
    crop_size = 24
    num_bins = 16
    batch_size = 64
    num_epochs = 5
    learning_rate = 0.001
    seed = 42
    cnn_hidden_channels = [8, 8, "A2", 16, 16, "A2", 32, 32, "A2", 64, 64, "AA"]
    use_noise = False

    main(
        crop_size=crop_size,
        num_bins=num_bins,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        cnn_hidden_channels=cnn_hidden_channels, 
        seed=seed,
        use_noise=use_noise,
    )
