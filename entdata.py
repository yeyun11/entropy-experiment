from scipy.ndimage import gaussian_filter1d
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from entcal import calculate_entropy
import numpy as np
from abc import ABC, abstractmethod


class BaseEntropyDataset(Dataset, ABC):
    """
    Base class for entropy datasets that handles common functionality.
    
    Args:
        dim (int): Dimension of each sample
        num_bins (int): Number of bins for entropy calculation
        seed (int, optional): Random seed for reproducibility
    """
    def __init__(self, dim=128, num_bins=10, seed=42):
        self.dim = dim
        self.num_bins = num_bins

        self._uniform_line_inc = torch.linspace(0, 1, dim)
        self._uniform_line_dec = torch.linspace(1, 0, dim)
        self._Dirac_like = torch.empty(dim)
        
        # Set seed for reproducibility if provided
        if seed is not None:
            self.seed = seed
            self.rng = torch.Generator()
            self.rng.manual_seed(seed)
        else:
            self.seed = None
            self.rng = None
    
    def _generate_single_sample(self):
        """
        Generate a single sample by randomly mixing a full entropy sample and a low entropy sample, then shuffle.
        
        Returns:
            torch.Tensor: A single generated sample
        """
        # Determine mix ratio and number of values to replace
        mix_ratio = torch.rand(1).item()
        dim_from_uniform = round(mix_ratio * self.dim)
        
        # Randomly mix uniform and Dirac with segments
        self._Dirac_like.fill_(torch.rand(1).item())
        
        # Equivalently flip
        sample = self._uniform_line_inc.clone() if torch.rand(1).item() < 0.5 else self._uniform_line_dec.clone()

        # Roll the sample to the left by a random number within [0, dim)
        roll_amount = torch.randint(0, self.dim, (1,)).item()
        sample = torch.roll(sample, roll_amount)
        
        # Replace selected indices with Dirac-like values
        sample[:dim_from_uniform] = self._Dirac_like[:dim_from_uniform]

        # Randomly shuffle to bring in high frequency confusion
        shuffle_amount = torch.randint(0, self.dim, (1,)).item()
        shuffle_indices = torch.randperm(shuffle_amount)
        sample[:shuffle_amount] = sample[shuffle_indices]

        # Randomly roll again
        roll_amount = torch.randint(0, self.dim, (1,)).item()
        sample = torch.roll(sample, roll_amount)

        # Randomly horizontal flip a random length segment
        flip_amount = torch.randint(1, self.dim, (1,)).item()
        sample[:flip_amount] = sample[:flip_amount].flip(0)

        # Randomly roll again
        roll_amount = torch.randint(0, self.dim, (1,)).item()
        sample = torch.roll(sample, roll_amount)

        return sample
    
    def _generate_samples(self, batch_size):
        return torch.stack([self._generate_single_sample() for _ in range(batch_size)])
    
    def _calculate_entropy(self, data):
        """
        Calculate entropy for the given data.
        
        Args:
            data (torch.Tensor): Data to calculate entropy for
            
        Returns:
            torch.Tensor: Entropy values
        """
        return calculate_entropy(data, 0, 1, self.num_bins)
    
    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass


class EntropyDataset(BaseEntropyDataset):
    """
    Dataset that generates random samples with a given dimension and calculates their entropy.
    This dataset generates all samples at initialization time (finite sample size).
    
    Args:
        num_samples (int): Number of samples to generate
        dim (int): Dimension of each sample
        num_bins (int): Number of bins for entropy calculation
        seed (int, optional): Random seed for reproducibility
    """
    def __init__(self, num_samples=1000, dim=100, num_bins=10, seed=None):
        super().__init__(dim, num_bins, seed)
        self.num_samples = num_samples
        
        # Generate data
        self.data = self._generate_samples(self.num_samples)
        
        # Calculate entropy labels
        self.labels = self._calculate_entropy(self.data)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class OnTheFlyEntropyDataset(BaseEntropyDataset):
    """
    Dataset that generates random samples on-the-fly with a given dimension and calculates their entropy.
    This dataset generates new samples each time __getitem__ is called (infinite sample size).
    
    Args:
        max_samples (int): Maximum number of samples to generate (for __len__ purposes)
        dim (int): Dimension of each sample
        num_bins (int): Number of bins for entropy calculation
        seed (int, optional): Random seed for reproducibility
    """
    def __init__(self, max_samples=1000000, dim=100, num_bins=10, seed=None):
        super().__init__(dim, num_bins, seed)
        self.max_samples = max_samples
    
    def __len__(self):
        return self.max_samples
    
    def __getitem__(self, idx):
        # Generate a new sample on-the-fly
        sample = self._generate_samples(1)
        # Calculate entropy for the sample
        entropy = self._calculate_entropy(sample)
        return sample.squeeze(0), entropy


def get_entropy_dataloader(num_samples=1000, dim=128, num_bins=10, 
                          batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader for entropy data with finite sample size.
    
    Args:
        num_samples (int): Number of samples to generate
        dim (int): Dimension of each sample
        num_bins (int): Number of bins for entropy calculation
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for the DataLoader
        
    Returns:
        DataLoader: DataLoader for the entropy dataset
    """
    dataset = EntropyDataset(
        num_samples=num_samples,
        dim=dim,
        num_bins=num_bins
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def get_train_test_dataloaders(num_samples=1000, dim=128, num_bins=10, 
                              test_ratio=0.2, batch_size=32, shuffle=True, 
                              num_workers=0, seed=42):
    """
    Create train and test DataLoaders for entropy data with train/test split.
    
    Args:
        num_samples (int): Total number of samples to generate
        dim (int): Dimension of each sample
        num_bins (int): Number of bins for entropy calculation
        test_ratio (float): Ratio of test samples (between 0 and 1)
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for the DataLoader
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Create the dataset
    dataset = EntropyDataset(
        num_samples=num_samples,
        dim=dim,
        num_bins=num_bins
    )
    
    # Calculate train and test sizes
    test_size = int(num_samples * test_ratio)
    train_size = num_samples - test_size
    
    # Split the dataset
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers
    )
    
    return train_dataloader, test_dataloader


def get_onthefly_dataloader(max_samples=1000000, dim=128, num_bins=10, 
                           batch_size=32, num_workers=0, seed=None):
    """
    Create a DataLoader for on-the-fly entropy data generation.
    
    Args:
        max_samples (int): Maximum number of samples (for __len__ purposes)
        dim (int): Dimension of each sample
        num_bins (int): Number of bins for entropy calculation
        batch_size (int): Batch size for the DataLoader
        num_workers (int): Number of workers for the DataLoader
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        DataLoader: DataLoader for on-the-fly entropy data generation
    """
    dataset = OnTheFlyEntropyDataset(
        max_samples=max_samples,
        dim=dim,
        num_bins=num_bins,
        seed=seed
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle as samples are generated randomly
        num_workers=num_workers
    )


if __name__ == "__main__":
    dataset = EntropyDataset(num_samples=10000, dim=128, num_bins=16)
    from matplotlib import pyplot as plt

    # plot the first 20 samples
    plt.figure(figsize=(20, 16))
    for i in range(1, 6):
        plt.subplot(5, 2, i)
        plt.plot(dataset.data[i-1])
        plt.title(f"Entropy: {dataset.labels[i].item():.4f}")
        plt.subplot(5, 2, i + 5)
        plt.plot(dataset.data[-i])
        plt.title(f"Entropy: {dataset.labels[-i].item():.4f}")
    plt.savefig("data_samples.png")
    plt.close()

    plt.figure()
    plt.hist(dataset.labels, 50)
    plt.savefig("data_entropy_dist.png")
    plt.close()
