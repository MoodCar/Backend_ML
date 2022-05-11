import torch
import torch.nn as nn


class SBERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=7, device='cuda:0', dr_rate=0.5):
        super(SBERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.device = device
        self.classifier = nn.Linear(hidden_size , num_classes).to(self.device)
        # nn.init.xavier_uniform_(self.classifier.weight, 0.0)
        if dr_rate is not None:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def forward(self, x):
        out = self.bert.encode(x, device=self.device, convert_to_tensor=True)
        if self.dr_rate:
            out = self.dropout(out)
            
        return out, self.classifier(out)
    
    
class KoBERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,
                 dr_rate=0.5,
                 ):
        super(KoBERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        # nn.init.xavier_uniform_(self.classifier.weight, 0.0)
        if dr_rate is not None:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        out = pooler
        return out, self.classifier(out)