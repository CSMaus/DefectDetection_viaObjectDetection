import json
import matplotlib.pyplot as plt

path = "models/HybridBinaryModel_20250718_1521/training_history.json"
with open(path, "r") as f:
    hist = json.load(f)

epochs = hist["epochs"]
train_loss = hist["train_loss"]
val_loss = hist["val_loss"]
train_acc = hist["train_accuracy"]
val_acc = hist["val_accuracy"]

plt.figure(figsize=(6,8))

# Accuracy subplot (top)
plt.subplot(2,1,1)
plt.plot(epochs, train_acc, label="Training", color="blue")
plt.plot(epochs, val_acc, label="Validation", color="red")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.grid(True)

# Loss subplot (bottom)
plt.subplot(2,1,2)
plt.plot(epochs, train_loss, label="Training", color="blue")
plt.plot(epochs, val_loss, label="Validation", color="red")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
