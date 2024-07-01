from torch import nn
from transformers import XLMRobertaConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel


class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    config_xlm = XLMRobertaConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False) # without pooling (eg. [CLS])
        self.clf = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)    
        )
        self.init_weights()
        self._loss = nn.CrossEntropyLoss()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        output = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        logits = self.clf(output[0])
        if labels is not None:
            loss = self._loss(logits.view((-1, self.num_labels)), labels.view(-1))
        else:
            loss = None
        return TokenClassifierOutput(loss=loss, logits=logits, 
                                     hidden_states=output.hidden_states, attentions=output.attentions)