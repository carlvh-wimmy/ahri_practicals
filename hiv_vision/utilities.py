from sklearn.metrics import precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

def precision(inp, targ):
    preds = inp.argmax(dim=1)
    return precision_score(targ, preds, average='weighted', zero_division=0)

def recall(inp, targ):
    preds = inp.argmax(dim=1)
    return recall_score(targ, preds, average='weighted', zero_division=0)

    
def plot_metrics(learn):
    # Calculate the average training loss per epoch
    num_batches_per_epoch = len(learn.recorder.losses) // len(learn.recorder.values)
    train_loss_per_epoch = [np.mean(learn.recorder.losses[i * num_batches_per_epoch:(i + 1) * num_batches_per_epoch]).item() 
                            for i in range(len(learn.recorder.values))]

    # Extract validation loss, accuracy, precision, and recall from learn.recorder.values
    valid_loss = [v[learn.recorder.metric_names.index('valid_loss') - 1] for v in learn.recorder.values]
    accuracy = [v[learn.recorder.metric_names.index('accuracy') - 1] for v in learn.recorder.values]
    # precision = [v[learn.recorder.metric_names.index('precision') - 1] for v in learn.recorder.values]
    # recall = [v[learn.recorder.metric_names.index('recall') - 1] for v in learn.recorder.values]

    # Create the first plot for training and validation loss
    epochs = range(len(valid_loss))  # valid_loss is calculated per epoch

    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)  # First subplot (1 row, 2 columns, 1st plot)
    plt.plot(epochs, train_loss_per_epoch, label='Training Loss', alpha=0.6)
    plt.plot(epochs, valid_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    # plt.plot(epochs, precision, label='Precision', color='green', marker='o')
    # plt.plot(epochs, recall, label='Recall', color='red', marker='s')
    plt.plot(epochs, accuracy, label='Accuracy', color='blue', marker='^')
    
    
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Accuracy and  Precision over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()