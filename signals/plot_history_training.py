import json
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman'})


def plot_training_history(filename):
    with open(filename, 'r') as f:
        history = json.load(f)

    epochs = history['epochs']
    train_loss = history['train_loss']

    # need to multiply by 100 to get percentage

    train_accuracy = [acc * 100 for acc in history['train_accuracy']]

    plt.figure(figsize=(6, 5))
    plt.plot(epochs, train_loss, label='Training Loss', color='red')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss value')
    # plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(epochs, train_accuracy, label='Training Accuracy', color='blue')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy, %')
    # plt.legend()
    plt.grid(True)
    plt.show()


plot_training_history('MultiSignalClassifier-training_history3.json')
