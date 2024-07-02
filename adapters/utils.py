from sklearn.metrics import f1_score, accuracy_score


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


def freeze_n_bottom_layers(model, n, freeze_embeedings=True):
    if n == -1:
        return model
    for layer_to_freeze in range(n):
        for name, param in model.named_parameters():
            if name.startswith(f'distilbert.transformer.layer.{layer_to_freeze}.'):
                param.requires_grad = False
    if freeze_embeedings:
        for name, param in model.named_parameters():
            if name.startswith('distilbert.embeddings.'):
                param.requires_grad = False
    return model