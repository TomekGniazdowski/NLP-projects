from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


def plot_confusion_matrix(y_true: list, y_pred: list):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm = np.round(cm, 3)
    _, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, ax=ax, cmap="YlGnBu")
    ax.set_title("Normalized confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.show()
    

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_micro = f1_score(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'acc': acc
    }