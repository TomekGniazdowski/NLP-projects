import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


class TransformerEncoderFromScratchForSequenceClassification(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self._transformer = TransformerEncoderFromScratch(config)
        self._clf = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        
    def forward(self, x):
        x = self._transformer(x)[:, 0, :]
        return self._clf(x)
        

class TransformerEncoderFromScratch(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self._embeddings = Embeddings(config.vocab_size, config.hidden_size, config.max_position_embeddings)
        self._encoder_layer = nn.ModuleList([
            EncoderLayer(config.hidden_size, config.intermediate_size, config.hidden_dropout_prob, config.num_attention_heads) for _ in range(config.num_hidden_layers)
        ])
        
    def forward(self, x):
        x = self._embeddings(x)
        for encoder_layer in self._encoder_layer:  
            x = encoder_layer(x)
        return x


class Embeddings(nn.Module):
    
    def __init__(self, vocab_size, hidden_dim, max_position_embeddings):
        super().__init__()
        
        self._token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self._position_embedding = nn.Embedding(max_position_embeddings, hidden_dim)
        self._layers = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout()
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, x):
        seq_len = x.shape[1]
        token_embedding = self._token_embedding(x)
        position_embedding = self._position_embedding(torch.arange(seq_len, dtype=torch.long, device=self._device).unsqueeze(0))
        return self._layers(token_embedding + position_embedding)


class EncoderLayer(nn.Module):
    
    def __init__(self, hidden_dim, intermediate_dim, hidden_dropout_prob, num_attention_heads):
        super().__init__()
        
        self._layer_norm_1 = nn.LayerNorm(hidden_dim)
        self._layer_norm_2 = nn.LayerNorm(hidden_dim)
        self._multi_head_attention = MultiHeadAttention(hidden_dim, num_attention_heads)
        self._feed_forward = FeedForward(hidden_dim, intermediate_dim, hidden_dropout_prob)
    
    def forward(self, x):
        x = x + self._multi_head_attention(self._layer_norm_1(x))
        return x + self._feed_forward(self._layer_norm_2(x))
        

class FeedForward(nn.Module):
    
    def __init__(self, hidden_dim, intermediate_dim, hidden_dropout_prob):
        super().__init__()
        
        self._layers = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(hidden_dropout_prob)
        )
    
    def forward(self, x):
        return self._layers(x)


class MultiHeadAttention(nn.Module):
    
    def __init__(self, embed_dim, num_attention_heads):
        super().__init__()
        
        head_dim = embed_dim // num_attention_heads
        self._multi_head_attention = nn.ModuleList([
            AttentionHead(embed_dim, head_dim) for _ in range(num_attention_heads)
        ])
        self._linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        output = torch.cat([head(x) for head in self._multi_head_attention], dim=-1)
        return self._linear(output)
        


class AttentionHead(nn.Module):
    
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        
        self._linear_query = nn.Linear(embed_dim, head_dim)
        self._linear_key = nn.Linear(embed_dim, head_dim)
        self._linear_value = nn.Linear(embed_dim, head_dim)
    
    def forward(self, x):
        return scaled_dot_product_attention(
            query=self._linear_query(x), 
            key=self._linear_key(x), 
            value=self._linear_value(x)
            )
    

def scaled_dot_product_attention(query, key, value):
    weights = F.softmax(torch.bmm(query, key.transpose(1, 2)) / sqrt(query.size(-1)), dim=-1)
    return torch.bmm(weights, value)