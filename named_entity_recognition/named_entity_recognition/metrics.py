import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from seqeval.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch.nn.functional import cross_entropy


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm = np.round(cm, 3)
    _, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, ax=ax, cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    ax.set_title("Normalized confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.show()


def predictions_labels_to_seqeval_format(predictions, labels, id2label):
    preds = np.argmax(predictions, axis=2)
    batch, _ = preds.shape
    preds_seqeval = [[] for _ in range(batch)]
    labels_seqeval = [[] for _ in range(batch)]
    for i in range(batch):
        for pred, label in zip(preds[i], labels[i]):
            if label != -100:
                preds_seqeval[i].append(id2label[pred])
                labels_seqeval[i].append(id2label[label])
    return preds_seqeval, labels_seqeval


def prepare_compute_metrics(id2label):
    def compute_metrics(pred, id2label=id2label):
        y_pred, y_true = predictions_labels_to_seqeval_format(pred.predictions, pred.label_ids, id2label)
        return {'f1': f1_score(y_true, y_pred)}
    return compute_metrics


def compute_predictions_loss(batch, trainer, data_collator, device):
    batch = data_collator([dict(zip(batch, v)) for v in zip(*batch.values())])
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        output = trainer.model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(output.logits, dim=2).cpu().numpy()
    loss = cross_entropy(output.logits.view((-1, 7)), labels.view(-1), reduction='none')
    return {'preds': preds, 'loss': loss.view((len(input_ids), -1)).cpu().numpy()}