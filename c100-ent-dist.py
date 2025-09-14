# plot entropy distribution of CIFAR-100
import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from entcal import calculate_entropy
import matplotlib.pyplot as plt

from skimage.restoration import non_local_means, estimate_sigma

use_noise = True

# Load CIFAR-100 dataset
transform = transforms.Compose([
    transforms.CenterCrop(24),
    transforms.ToTensor(),
])
cifar100_test = datasets.CIFAR100(root='./cifar-100', train=False, download=True, transform=transform)

# iterate all the images in the dataset
entropies = []
for image, _ in tqdm(cifar100_test):
    if use_noise:
        image = image.numpy()
        sigma = estimate_sigma(image, average_sigmas=True)
        denoised = non_local_means.denoise_nl_means(image, sigma=sigma, h=0.025, channel_axis=0)
        noise = image - denoised
        noise = torch.from_numpy(noise)
        #print(noise.min().item(), noise.max().item())
        entropy = calculate_entropy(noise[None, ], min_value=noise.min(), max_value=noise.max())
    else:
        entropy = calculate_entropy(image[None, ])
    entropies.append(entropy.item())

# Plot histogram
plt.hist(entropies, bins=50)
plt.title('Entropy Distribution of CIFAR-100 Images')
plt.xlabel('Entropy')
plt.ylabel('Frequency')
tag_use_noise = "noise" if use_noise else "image"
plt.savefig('cifar-100-entropy-dist-{}.png'.format(tag_use_noise))
plt.close()
