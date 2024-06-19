import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns
import fasttext


def vectorize_text(X: torch.Tensor, y:  torch.Tensor, ft_model, max_num_of_words: int):
    X_vec = []
    labels = []

    for text, label in zip(X, y):
        text_vectorized = np.array([ft_model.get_word_vector(word) for 
                                    word in fasttext.tokenize(text)[:max_num_of_words]])
        if len(text_vectorized) > 0:
            X_vec.append(torch.from_numpy(text_vectorized))
            labels.append(label)

    X_vec = pad_sequence(X_vec, batch_first=True)
    assert len(X_vec)==len(labels)
    dataset_vec = tuple(zip(X_vec, labels))

    return dataset_vec


def plot_confusion_matrix(y_true: list, y_pred: list):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm = np.round(cm, 3)
    _, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, ax=ax, cmap="YlGnBu")
    ax.set_title("Normalized confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.show()