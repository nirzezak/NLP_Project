from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

import config
from config import DEVICE


class SARCBert(nn.Module):
    def __init__(self, bert_model, config):
        super(SARCBert, self).__init__()
        self.bert = bert_model
        self.run_config = config

        if self.run_config.use_ancestors:
            in_size = self.run_config.max_ancestors * self.bert.config.hidden_size
            out_size = self.bert.config.hidden_size
            self.ancestor_layer = nn.LSTM(input_size=self.bert.config.hidden_size,
                                          hidden_size=self.bert.config.hidden_size, num_layers=2, batch_first=False,
                                          dropout=0.1, bidirectional=True)
            # self.ancestor_layer = nn.Linear(in_features=in_size, out_features=out_size)
            # self.ancestor_activation_layer = self.run_config.activation_func
            self.classifier = nn.Linear(in_features=self.bert.config.hidden_size * 3, out_features=2)
        else:
            self.classifier = nn.Linear(in_features=self.bert.config.hidden_size, out_features=2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, ancestor_input_ids=None,
                ancestor_attention_mask=None, labels=None, **kwargs):
        input_bert_res = self.bert(input_ids, attention_mask)
        input_cls_tokens = input_bert_res.last_hidden_state[:, 0, :]

        # Handle the case of no ancestors:
        if not self.run_config.use_ancestors:
            res = self.classifier(input_cls_tokens)
            loss = self.loss(res, labels)
            return {'loss': loss, 'logits': res}

        ancestor_bert_res = []
        for i in range(self.run_config.max_ancestors):
            ancestor_bert_res.append(self.bert(ancestor_input_ids[i], ancestor_attention_mask[i]))

        ancestor_cls_tokens = []
        for i in range(self.run_config.max_ancestors):
            # Use DAN as a way to represent a sentence
            tokens = ancestor_bert_res[i].last_hidden_state
            tokens_sum = tokens.sum(dim=1)
            active_tokens = ancestor_attention_mask[i].sum(dim=1)
            tokens_avg = tokens_sum / active_tokens.unsqueeze(dim=1)
            ancestor_cls_tokens.append(tokens_avg)
            # ancestor_cls_tokens.append(ancestor_bert_res[i].last_hidden_state[:, 0, :])

        # ancestor_cls_tokens = torch.cat(ancestor_cls_tokens, dim=1)
        # ancestor_cls_tokens = self.ancestor_layer(ancestor_cls_tokens)
        # ancestor_cls_tokens = self.ancestor_activation_layer(ancestor_cls_tokens)

        ancestor_cls_tokens = torch.stack(ancestor_cls_tokens)
        lstm_output, _ = self.ancestor_layer(ancestor_cls_tokens)
        final_lstm_state = lstm_output[-1]

        concat_cls_tokens = torch.cat([input_cls_tokens, final_lstm_state], dim=1).to(DEVICE)
        res = self.classifier(concat_cls_tokens)
        loss = self.loss(res, labels)
        return {'loss': loss, 'logits': res}


@dataclass
class SARCBertDataCollator:
    tokenizer: Any
    padding = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        c = config.get_config()

        # Handle inputs
        input_ids = [x.pop('input_ids') for x in features]
        padded_input_ids = self.tokenizer.pad({'input_ids': input_ids})

        # Handle labels
        if 'label' in features[0]:
            labels = torch.tensor([x.pop('label') for x in features]).long()
        else:
            labels = torch.tensor([x.pop('labels') for x in features]).long()

        # Create the batch
        batch = {
            'labels': labels,
            'input_ids': torch.tensor(padded_input_ids['input_ids']).long(),
            'attention_mask': torch.tensor(padded_input_ids['attention_mask']).long()
        }

        # Handle ancestors, if necessary
        if c.use_ancestors:
            ancestor_input_ids = [x.pop('ancestor_input_ids') for x in features]
            padded_ancestor_ids = self._pad_ancestors(ancestor_input_ids)

            batch['ancestor_input_ids'] = torch.tensor(padded_ancestor_ids['input_ids']).long()
            batch['ancestor_attention_mask'] = torch.tensor(padded_ancestor_ids['attention_mask']).long()

        return batch

    def _pad_ancestors(self, ancestor_input_ids):
        c = config.get_config()

        flattened_ancestor_input_ids = []
        for ancestors in ancestor_input_ids:
            for ancestor in ancestors:
                flattened_ancestor_input_ids.append(ancestor)

        padded_ancestor_ids = self.tokenizer.pad({'input_ids': flattened_ancestor_input_ids})
        padded_ancestor_ids['input_ids'] = torch.tensor(padded_ancestor_ids['input_ids']).reshape(c.max_ancestors,
                                                                                                  len(ancestor_input_ids),
                                                                                                  -1).tolist()
        padded_ancestor_ids['attention_mask'] = torch.tensor(padded_ancestor_ids['attention_mask']).reshape(
            c.max_ancestors, len(ancestor_input_ids), -1).tolist()

        return padded_ancestor_ids
