import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from entdata import get_train_test_dataloaders
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_mlp(input_dim=128, hidden_dims=[256, 256, 256]):
    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
        ])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, 1))
    
    return nn.Sequential(*layers)


def get_transformer(input_dim=128, num_heads=4, num_layers=2):
    return nn.Sequential(
        nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        ), 
        nn.Linear(input_dim, 1)
    )


def train_epoch(model, train_loader, criterion, optimizer, device):
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
    
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        
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


def evaluate(model, test_loader, criterion, device):
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
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
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
    
    plt.show()


def main(dim, num_samples, num_bins, batch_size, num_epochs, learning_rate, test_ratio, model_type, seed, model_params):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create train and test dataloaders
    print("Creating dataloaders...")
    train_loader, test_loader = get_train_test_dataloaders(
        num_samples=num_samples,
        dim=dim,
        num_bins=num_bins,
        test_ratio=test_ratio,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        seed=seed
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    if model_type == 'mlp':
        model = get_mlp(**model_params).to(device)
    elif model_type == 'transformer':
        model = get_transformer(**model_params).to(device)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    print("Starting training...")
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, predictions, targets = evaluate(model, test_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Log metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
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
    plot_results(predictions, targets, 'entropy_prediction_results-synth-{}.png'.format(model_type))
    
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
    plt.savefig('training_curves-synth-{}.png'.format(model_type), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training completed!")


if __name__ == "__main__":

    # Training parameters
    num_samples = 10000
    dim = 128
    num_bins = 16
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    test_ratio = 0.15
    seed = 42

    model_type = 'mlp'
    model_params = {
        'input_dim': dim,
        'hidden_dims': [256, 256, 256],
    }
    
    """
    model_type = 'transformer'
    model_params = {
        'input_dim': dim,
        'num_heads': 8,
        'num_layers': 4,
    }
    """

    main(
        dim=dim,
        num_samples=num_samples,
        num_bins=num_bins,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        test_ratio=test_ratio,
        model_type=model_type,
        seed=seed,
        model_params=model_params,
    )
