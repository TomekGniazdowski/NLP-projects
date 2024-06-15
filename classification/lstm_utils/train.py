import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from lstm_utils.utils import plot_confusion_matrix


def test(model: nn.Module, dataloader: DataLoader, device: str):
    y_true = []
    y_preds = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch.to(device))
            preds = torch.argmax(y_pred, dim=1)
            y_true += y_batch.tolist()
            y_preds += preds.tolist()
    
    plot_confusion_matrix(y_true=y_true, y_pred=y_preds)
    return classification_report(y_true=y_true, y_pred=y_preds, output_dict=True)


def validate(
    model: nn.Module,
    loss_fn: nn.CrossEntropyLoss,
    dataloader: DataLoader,
    device: str
):
    loss = 0
    batch_ctr = 0
    y_true = []
    y_preds = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch.to(device))
            batch_ctr += len(y_pred)
            loss += loss_fn(y_pred, y_batch.to(device)).sum()
            preds = torch.argmax(y_pred, dim=1)
            y_true += y_batch.tolist()
            y_preds += preds.tolist()

    f1 = f1_score(y_true=y_true, y_pred=y_preds, average='macro')
    return loss / batch_ctr, f1


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.CrossEntropyLoss,
    epochs: int,
    train_dl: DataLoader,
    val_dl: DataLoader,
    patience: int,
    print_metrics: bool,
    device: str,
    best_model_path: str
):

    fit_logs_train = {'loss': [], 'f1': []}
    fit_logs_val = {'loss': [], 'f1': []}
    best_val_f1 = 0
    patience_counter = 0
    pbar = tqdm(range(epochs))
    
    for epoch in pbar:
        model.train()
        for X_batch, y_batch in train_dl:
            y_pred = model(X_batch.to(device))
            loss = loss_fn(y_pred, y_batch.to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        train_loss, train_f1 = validate(model=model, loss_fn=loss_fn, dataloader=train_dl, device=device)
        val_loss, val_f1 = validate(model=model, loss_fn=loss_fn, dataloader=val_dl, device=device)
        
        if print_metrics:
            pbar.set_postfix({
                'Epoch': epoch,
                'train loss': train_loss,
                'train f1': train_f1,
                'validation loss': val_loss,
                'validation f1': val_f1
                })

        fit_logs_train['f1'].append(train_f1)
        fit_logs_train['loss'].append(train_loss)

        fit_logs_val['f1'].append(val_f1)
        fit_logs_val['loss'].append(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"New best model on epoch {epoch}, f1={best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping on epoch {epoch}")
                break
    
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(fit_logs_train['loss'], label='train')
    ax[0].plot(fit_logs_val['loss'], label='validation')
    ax[0].set_title("Loss")
    ax[0].legend()
    
    ax[1].plot(fit_logs_train['f1'], label='train')
    ax[1].plot(fit_logs_val['f1'], label='validation')
    ax[1].set_title("F1")
    ax[1].legend()
    plt.show()
    
    return model