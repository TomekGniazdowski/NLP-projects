import pandas as pd
import torch


def tag_text(text, tags, model, tokenizer, device):
    text_tokenized = tokenizer(text, return_tensors='pt')
    input_ids = text_tokenized['input_ids'].to(device)
    output = model(input_ids)
    preds = [tags.names[pred] for pred in torch.argmax(output.logits, dim=2)[0]]
    return pd.DataFrame({
        'tokens': text_tokenized.tokens(),
        'input_ids': input_ids[0].cpu().numpy(),
        'preds': preds
    }).T