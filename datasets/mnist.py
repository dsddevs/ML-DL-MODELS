from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Scales pixel values from [0, 255] to [0.0, 1.0] and converts to Tensor
pixel_to_tensor = transforms.ToTensor()

# Centers data around 0 with standard deviation of 1
# Accelerates convergence by stabilizing the gradient updates
normalization = transforms.Normalize(mean=(0.1307,), std=(0.3081,))

# === MNIST DATA PREPARATION ===
transform = transforms.Compose([pixel_to_tensor, normalization])

# MNIST train datasets (handwritten digits)
train_dataset = datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=transform
)

# MNIST test datasets (handwritten digits)
test_dataset = datasets.MNIST(
    './data',
    train=False,
    transform=transform
)


# === MNIST DATA LOADER ===
# Manage batching (e.g., 64 images per step) and shuffling for the training process
# Batching allows for parallel processing on GPU/CPU

def get_data_for_training():
    return DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )


def get_data_for_testing():
    return DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )

