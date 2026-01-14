import torch.nn as nn


# Simple Convolutional Neural Network (CNN) architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # ========================== CONVOLUTIONAL FEATURE EXTRACTION ===========================
        # This block extracts spatial features (like edges and shapes) from the images.
        self.conv_layers = nn.Sequential(

            # ========================== CNN LAYER - 1 ===============================
            # in_channels: 1 (Grayscale input, 1 intensity value per pixel)
            # out_channels: 32 (Number of different filters/features to learn)
            # kernel_size: 3 (3x3 sliding window to scan the image)
            # padding: 1 (Preserves the original spatial dimensions)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),

            # Activation function: Replaces negative values with zero (Rectified Linear Unit)
            # Introduces non-linearity, allowing the network to learn complex patterns
            nn.ReLU(),

            # Pooling: Reduces spatial resolution (height and width) by 2x
            # Retains the most significant features and reduces computational load
            nn.MaxPool2d(2),

            # ========================== CNN LAYER - 2 ===============================
            # in_channels: 32 (Receives 32 feature maps from the previous layer)
            # out_channels: 64 (Extracts more complex combinations of features)
            # kernel_size: 3, padding: 1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            # Non-linear activation to deepen feature representation
            nn.ReLU(),

            # Further reduces dimensions by 2x (7x7 spatial resolution for MNIST)
            nn.MaxPool2d(2),
        )

        # ========================== FULLY CONNECTED CLASSIFIER ===============================
        # This block interprets the extracted features and maps them to specific classes.
        self.fc_layers = nn.Sequential(

            # Flattens 2D feature maps (64 channels of 7x7) into a 1D vector (3136 elements)
            # Necessary for transitioning from convolutional layers to linear layers
            nn.Flatten(),

            # === LINEAR LAYER (FULLY CONNECTED) ===
            # Maps 3136 input features to 128 hidden neurons
            nn.Linear(64 * 7 * 7, 128),

            # Non-linear activation for high-level abstraction
            nn.ReLU(),

            # Dropout: Randomly deactivates 50% of neurons during training
            # A regularization technique used to prevent overfitting and improve generalization
            nn.Dropout(0.5),

            # === OUTPUT LAYER ===
            # Maps 128 neurons to 10 output scores (one for each digit 0-9)
            nn.Linear(128, 10)
        )

    def forward(self, x):
        """ Defines the forward pass of the model: input -> convolution -> classification """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
