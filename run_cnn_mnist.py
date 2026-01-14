import torch
import torch.nn as nn
import torch.optim as optim

from cnn.cnn_mnist_01 import SimpleCNN
from datasets.mnist import get_data_for_training, get_data_for_testing

data_for_training = get_data_for_training()
data_for_testing = get_data_for_testing()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SimpleCNN().to(device)
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === MODEL TRAINING ===
epochs = 5
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()
    total_error_outputs = 0.0
    total_correct_outputs = 0
    total_labels = 0

    for batch_idx, (data, labels) in enumerate(data_for_training):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()                          # grads to None
        prediction = model(data)
        loss = cross_entropy_loss(prediction, labels)  # predicted_outputs == labels (comparing)
        loss.backward()                                # gradient descent: gradient computing (output -> input) to minimize loss
        optimizer.step()                               # updated weights

        # === REPORT METRICS ===
        total_error_outputs += loss.item()
        _, idx_max_predicted = prediction.max(1)       # [id_max: 10.5, id_max: 8.9, id_max:4.3] > [0, 1, 1]
        total_labels += labels.size(0)
        total_correct_outputs += idx_max_predicted.eq(labels).sum().item()

    # === EPOCH METRICS ===
    epoch_loss \
        = total_error_outputs / len(data_for_training)
    epoch_acc = 100. * total_correct_outputs / total_labels
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

# === MODEL TESTING ===
model.eval()                                             # evaluation mode (deactivates Dropout)
test_loss = 0
total_correct_outputs = 0
total_labels = 0

with torch.no_grad():
    for data, labels in data_for_testing:
        data, labels = data.to(device), labels.to(device)
        prediction = model(data)
        test_loss += cross_entropy_loss(prediction, labels).item()
        _, idx_max_predicted = prediction.max(1)
        total_labels += labels.size(0)
        total_correct_outputs += idx_max_predicted.eq(labels).sum().item()

test_loss /= len(data_for_testing)
test_accuracy = 100. * total_correct_outputs / total_labels

# === PRINT STATISTICS ===
print(f"\n === TEST RESULT ====")
print(f"Average Loss: {test_loss:.4f}")
print(f"Accuracy: {test_accuracy:.2f}%")

# === SAVING MODEL WEIGHTS ===
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print("Model saved as 'mnist_cnn_model.pth'")

