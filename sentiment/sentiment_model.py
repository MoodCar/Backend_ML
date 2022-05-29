import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np


from sentence_transformers import SentenceTransformer
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
from utils.dataset import KoBERTDataset
from utils.classifier import KoBERTClassifier
from utils.classifier import SBERTClassifier

from iteround import saferound
# from sklearn.model_selection import train_test_split

from kss import split_sentences
from konlpy.tag import Okt
                                               

                                                         
import random


MODELS = [
    'KoBERT',
    'SBERT',
]


def get_sentiment_model_class(model_name):
    """Return the algorithm class with the given name."""
    if model_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(model_name))
    return globals()[model_name]


class SentimentModel:
    def __init__(self, cfg):
        self.label_name = cfg['label_name']
        self.num_classes = len(self.label_name)
        self.device = cfg['device']
    
    
    def predict(self, x):
        raise NotImplementedError



class KoBERT(SentimentModel):
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
            cfg['label_name'](list(str)) : label name
            cfg['num_wokrers'](int) : num_workers
        
        """
        super(KoBERT, self).__init__(cfg)
        bertmodel, self.vocab = get_pytorch_kobert_model(cachedir=".cache")
        tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(tokenizer, self.vocab, lower=False)
        
        self.device = cfg['device']
        
        self.max_len = cfg['max_len']
        self.batch_size = cfg['batch_size']
        
        self.num_workers = cfg['num_workers']
        
        self.model = KoBERTClassifier(bertmodel).to(self.device)
        if cfg['model_path'] is not None:
            self.model.load_state_dict(torch.load(cfg['model_path'], map_location=self.device))
            
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
        
        parsing_sentences = split_sentences(x, num_workers=self.num_workers)
        
        x = KoBERTDataset(parsing_sentences, self.tok, self.max_len)
        dataloader = DataLoader(x, batch_size = self.batch_size, num_workers=self.num_workers)
        
        with torch.no_grad():
          total_out = torch.zeros(self.num_classes).to(self.device)
          
          for token_ids, valid_length, segment_ids in dataloader:
              token_ids = token_ids.long().to(self.device)
              segment_ids = segment_ids.long().to(self.device)
              
              _, out = self.model(token_ids, valid_length, segment_ids) 
              total_out += out.mean(dim=0)
              out = nn.Softmax(dim=1)(out)


        
        total_out /= len(dataloader)
        total_out = nn.Softmax(dim=0)(total_out)
        total_out = total_out * 100
        
        score_label = ['fear_score', 'suprise_score', 'anger_score', 'sad_score', 'neutral_score', 'happy_score', 'disgust_score']

        tmp_score = [total_out[i].float().item() for i in range(self.num_classes)]
        tmp_score = saferound(tmp_score, 1)
        
        predict_emotion = self.label_name[total_out.argmax()]
        predict_score = {score_label[i] : tmp_score[i] for i in range(self.num_classes)}
        
        
        
    
        return predict_score, predict_emotion
    
    
class SBERT(SentimentModel):
    def __init__(self, cfg):
        super(SBERT, self).__init__(cfg)
        # self.batch_size = cfg['batch_size']
        bert = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
        
        self.model = SBERTClassifier(bert, num_classes=self.num_classes, device=self.device)
        self.num_workers = cfg['num_workers']
        self.okt = Okt()
        
        self.model.eval()
        
    def predict(self, x):
        
        
        x = self.okt.normalize(x)
        x = split_sentences(x, num_workers=self.num_workers)
  
        with torch.no_grad():
            _,out = self.model(x)
        
  
        total_out = out.sum(dim=0)
        
        predict_emotion = self.label_name[total_out.argmax()]
        
        return predict_emotion
