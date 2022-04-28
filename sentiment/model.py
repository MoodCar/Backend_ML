import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np

from dataset import BERTDataset

from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
# from sklearn.model_selection import train_test_split

from kss import split_sentences
from konlpy.tag import Okt
                                               

                                                         
import random

class Model:
    """
    일기 감정 분석하는 모델
    
    Attributes:
        tok : KoBERT tokenizer
        vocab : KoBERT vocab
        device : model device
        num_classes : 분류하고자 하는 감정 개수
        max_len : input 최대 길이
        batch_size : batch size
        label_name : 감정 종류
        model : KoBERT 모델
        okt : 오타수정 해주는 라이브러리(속도는 느림)
    
    """
    def __init__(self, cfg):
        """
        KoBERT 모델 설정하는 클래스
        
        Args
            cfg['batch_size'](int) : batch_size
            cfg['model_path'](str) : 모델 파라미터 파일 경로
            cfg['device'](str) : device(gpu or cpu)
            cfg['max_len'](int) : KoBERT input 사이즈
            cfg['label_name'](list(str)) : 
        
        """
        bertmodel, self.vocab = get_pytorch_kobert_model(cachedir=".cache")
        tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(tokenizer, self.vocab, lower=False)
        
        self.device = cfg['device']
        self.label_name = cfg['label_name']
        self.num_classes = len(self.label_name)
        
        self.max_len = cfg['max_len']
        self.batch_size = cfg['batch_size']
        
        self.model = BERTClassifier(bertmodel).to(self.device)
        if cfg['model_path'] is not None:
            self.model.load_state_dict(torch.load(cfg['model_path']), map_location=self.device)
            
        self.model.eval()
        
        self.okt = Okt()
        
        

    
    def predict(self, x):
        """
        일기 내용을 바탕으로 감정을 반환하는 함수
        
        Args:
            x (str) : Diary Content, 
            
        Returns:
            str : Diary Emotion
        """
        
        
        # 오타 수정(필요할 경우에)
        x = self.okt.normalize(x)
        
        x = split_sentences(x)
        
        x = BERTDataset(x, self.tok, self.max_len)
        dataloader = DataLoader(x, batch_size = self.batch_size)
        
        with torch.no_grad():
          total_out = torch.zeros(self.num_classes)
          for token_ids, valid_length, segment_ids in dataloader:
              token_ids = token_ids.long().to(self.device)
              segment_ids = segment_ids.long().to(self.device)
              
              out = self.model(token_ids, valid_length, segment_ids) 
              total_out += out.sum(dim=0)
        
        
        predict_emotion = self.label_name[total_out.argmax()]
    
    
        return predict_emotion
    
    
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,
                 dr_rate=None,
                 ):
        super(BERTClassifier, self).__init__()
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
        return self.classifier(out)
    