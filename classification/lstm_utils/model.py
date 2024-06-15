import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers_lstm: int,
        dropout_lstm: float,
        dropout_clf: float
        ):
        
        super().__init__()

        self._lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers_lstm,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_lstm
            )
        self._clf = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Dropout(dropout_clf),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(dropout_clf),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Dropout(dropout_clf),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self._lstm(x)
        return self._clf(lstm_out[:, -1])


class BiLSTM(nn.Module):
    
    """ https://discuss.pytorch.org/t/how-to-make-an-lstm-bidirectional/142928 """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers_lstm: int,
        dropout_lstm: float,
        dropout_clf: float
        ):
        
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers_lstm = num_layers_lstm
        
        self._lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers_lstm,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_lstm
            )
        self._clf = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(True),
            nn.Dropout(dropout_clf),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(dropout_clf),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Dropout(dropout_clf),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        _, hidden_state = self._lstm(x)
        final_state = hidden_state[0].view(self.num_layers_lstm, 2, batch_size, self.hidden_dim)[-1]
        final_hidden_state = torch.cat((final_state[0], final_state[1]), 1)
        return self._clf(final_hidden_state)