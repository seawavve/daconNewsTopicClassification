import torch
import torch.nn as nn


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,
                 dr_rate=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    @staticmethod
    def gen_attention_mask(token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        result = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.float().to(token_ids.device)
        )
        _, out = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.float().to(token_ids.device)
        ).values()
        if self.dr_rate:
            out = self.dropout(out)
        return self.classifier(out)  # todo: softmax output


class BERTEnsemble(BERTClassifier):
    def __init__(self, bert, paths, device):
        super(BERTEnsemble, self).__init__(bert)
        self.models = []
        for path in paths:
            model = BERTClassifier(bert).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            self.models += (model,)

    def forward(self, token_ids, valid_length, segment_ids):
        averaged_output = 0
        num_models = len(self.models)
        for model in self.models:
            out = model(token_ids, valid_length, segment_ids)
            averaged_output += out / num_models

        return averaged_output