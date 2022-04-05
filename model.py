from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

DEVICE = torch.device('cuda:0')

class SARCBert(nn.Module):
    def __init__(self, bert_model):
        super(SARCBert, self).__init__()
        self.bert = bert_model

        self.ancestor_layer = nn.Linear(in_features=3 * self.bert.config.hidden_size, out_features=self.bert.config.hidden_size)

        self.classifier = nn.Linear(in_features=self.bert.config.hidden_size * 2, out_features=2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, ancestor_input_ids,
                ancestor_attention_mask, labels=None, **kwargs):
        input_bert_res = self.bert(input_ids, attention_mask)

        ancestor_bert_res = []
        for i in range(3):
            ancestor_bert_res.append(self.bert(ancestor_input_ids[i], ancestor_attention_mask[i]))

        input_cls_tokens = input_bert_res.last_hidden_state[:, 0, :]

        ancestor_cls_tokens = []
        for i in range(3):
            ancestor_cls_tokens.append(ancestor_bert_res[i].last_hidden_state[:, 0, :])

        ancestor_cls_tokens = torch.cat(ancestor_cls_tokens, dim=1)
        ancestor_cls_tokens = self.ancestor_layer(ancestor_cls_tokens)

        concat_cls_tokens = torch.cat([input_cls_tokens, ancestor_cls_tokens], dim=1).to(DEVICE)
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
        batch = {}

        input_ids = [x.pop('input_ids') for x in features]
        ancestor_input_ids = [x.pop('ancestor_input_ids') for x in features]

        padded_input_ids = self.tokenizer.pad({'input_ids': input_ids})

        flattened_ancestor_input_ids = []
        for ancestors in ancestor_input_ids:
            for ancestor in ancestors:
                flattened_ancestor_input_ids.append(ancestor)

        padded_ancestor_ids = self.tokenizer.pad({'input_ids': flattened_ancestor_input_ids})
        padded_ancestor_ids['input_ids'] = torch.tensor(padded_ancestor_ids['input_ids']).reshape(3, len(ancestor_input_ids), -1).tolist()
        padded_ancestor_ids['attention_mask'] = torch.tensor(padded_ancestor_ids['attention_mask']).reshape(3, len(ancestor_input_ids), -1).tolist()


        # padded_ancestor_ids = self.tokenizer.pad({'input_ids': ancestor_input_ids})

        if 'label' in features[0]:
            labels = torch.tensor([x.pop('label') for x in features]).long()
        else:
            labels = torch.tensor([x.pop('labels') for x in features]).long()

        batch['labels'] = labels
        batch['input_ids'] = torch.tensor(padded_input_ids['input_ids']).long()
        batch['attention_mask'] = torch.tensor(padded_input_ids['attention_mask']).long()
        batch['ancestor_input_ids'] = torch.tensor(padded_ancestor_ids['input_ids']).long()
        batch['ancestor_attention_mask'] = torch.tensor(padded_ancestor_ids['attention_mask']).long()

        return batch

    def _pad_ancestors(self, ancestor_ids):
        # Start with batch_size X max_ancestors X input_ids
        # End with max_ancestors X batch_size X padded_input_ids (per ancestor index)

        batch_size = len(ancestor_ids)
        max_ancestors = len(ancestor_ids[0])

        ancestors = [[]] * max_ancestors

        for ancestor in ancestor_ids:
            for i in range(max_ancestors):
                ancestors[i].append(ancestor[i])

        padded_ancestor_ids = []
        for i in range(max_ancestors):
            padded = self.tokenizer.pad({'input_ids': ancestors[i]})
        # TODO
