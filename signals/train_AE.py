import numpy as np
import os
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, save_model

healthy_signals_dir = 'D:/DataSets/!!NaWooDS/Health/'


def load_signals_from_txt(directory):
    signals = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        #  "Reference" should not be in the filename
        if 'Reference' not in filename and filename.endswith('.txt') and os.path.isfile(file_path):
            signal = np.loadtxt(file_path)
            signals.append(signal)
    return np.array(signals)


healthy_signals = load_signals_from_txt(healthy_signals_dir)

signal_length = healthy_signals.shape[1]
input_signal = Input(shape=(signal_length,))
encoded = Dense(64, activation='relu')(input_signal)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(signal_length, activation='linear')(decoded)

autoencoder = Model(input_signal, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(healthy_signals, healthy_signals, epochs=100, batch_size=32, shuffle=True, verbose=1)

save_model(autoencoder, 'autoencoder_model.h5')
print("Model trained and saved as 'autoencoder_model.h5'")
