import json
import matplotlib.pyplot as plt
import numpy as np

history_file = '../improved_multisignal/models/HybridBinaryModel_20250718_2100/training_history.json'
with open(history_file, 'r') as f:
    history = json.load(f)

# Limit to 15 epochs
epochs = history['epochs'][:15]
train_loss = history['train_loss'][:15]
train_accuracy = [acc * 100 for acc in history['train_accuracy'][:15]]  # Convert to percentage
val_loss = history['val_loss'][:15]
val_accuracy = [acc * 100 for acc in history['val_accuracy'][:15]]  # Convert to percentage

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Accuracy plot (top)
ax1.plot(epochs, train_accuracy, 'b-', linewidth=2, label='Training Accuracy')
ax1.plot(epochs, val_accuracy, 'g-', linewidth=2, label='Validation Accuracy')
ax1.set_ylabel('Accuracy (%)', fontsize=17)
ax1.set_ylim([94, 100])
ax1.grid(True, alpha=0.85)
ax1.tick_params(labelsize=15)
ax1.legend(fontsize=15)

# Loss plot (bottom)
ax2.plot(epochs, train_loss, 'r-', linewidth=2, label='Training Loss')
ax2.plot(epochs, val_loss, 'orange', linewidth=2, label='Validation Loss')
ax2.set_xlabel('Epoch', fontsize=17)
ax2.set_ylabel('Loss', fontsize=17)
ax2.grid(True, alpha=0.85)
ax2.tick_params(labelsize=15)
ax2.legend(fontsize=15)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()
