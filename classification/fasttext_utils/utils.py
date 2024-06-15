import fasttext
import fasttext.util
import re

def reduce_dim_fasttext(ft_model, dim: int, save_path: str):
    fasttext.util.reduce_model(ft_model, dim)
    ft_model.save_model(save_path)

def text_preprocessing(text: str):
    text = re.sub(r'http\S+', '', text) # delete URLs
    text = re.sub(r'@\S+', '', text) # delete mentions
    text = re.sub(r'#\S+', '', text) # delete hashtags
    text = re.sub(r'[^\w\s]', '', text) # delete punctuations
    text = text.lower()
    return text
    