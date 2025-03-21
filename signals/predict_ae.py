import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

autoencoder = load_model('autoencoder_model.h5')

test_signals_dir = 'path/to/test/signals'

test_signals = load_signals_from_txt(test_signals_dir)

reconstructed_signals = autoencoder.predict(test_signals)

reconstruction_errors = np.mean((test_signals - reconstructed_signals) ** 2, axis=1)

threshold = np.percentile(reconstruction_errors, 90)  # Top 10% of errors as defective

healthy_indices = np.where(reconstruction_errors <= threshold)[0]
defective_indices = np.where(reconstruction_errors > threshold)[0]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Healthy Signal Example")
plt.plot(test_signals[healthy_indices[0]], label='Original', alpha=0.6)
plt.plot(reconstructed_signals[healthy_indices[0]], label='Reconstructed', alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Defective Signal Example")
plt.plot(test_signals[defective_indices[0]], label='Original', alpha=0.6)
plt.plot(reconstructed_signals[defective_indices[0]], label='Reconstructed', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()

print("Reconstruction Errors:")
print("Healthy Example:", reconstruction_errors[healthy_indices[0]])
print("Defective Example:", reconstruction_errors[defective_indices[0]])
