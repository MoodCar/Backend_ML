from kss import split_sentences

import gluonnlp as nlp
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

from utils.classifier import KoBERTClassifier
from utils.classifier import SBERTClassifier
from utils.dataset import KoBERTDataset

import torch.nn as nn                                                                                 
import torch

from torch.utils.data import DataLoader
# class KeyWordExtractor:
#     def __init__(self,min_count=5, max_length=10, keyword_number=3, max_iter=10, beta=0.85):
#         self.wordrank_extractor = KRWordRank(
#             min_count=min_count,
#             max_length=max_length,
#             verbose=True
#         )
        
#         self.beta = beta
#         self.max_iter = max_iter
#         self.keyword_number = keyword_number
    
#     def extraction(self, contents):
#         contents = normalize(contents, english=True, number=True)
#         contents = split_sentences(contents, num_workers=1)
        
#         keywords, _, _ = self.wordrank_extractor.extract(contents, self.beta, self.max_iter,self.keyword_number)
       
#         return list(keywords.keys()


Models = [
    'TfIdf',
    'KoBERT',
    'SBERT',
]


def get_keyword_model_class(model_name):
    """Return the algorithm class with the given name."""
    if model_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(model_name))
    return globals()[model_name]

class KeywordModel:
    def __init__(self, cfg):
        self.max_keywords = cfg['max_keywords']
        
    def predict(self, x):
        raise NotImplementedError
    
    

class TfIdf(KeywordModel):
    """
    Agrs:
    
    """
    def __init__(self, cfg):
        """  
        Args
        """
        super().__init__(cfg)
        self.okt = Okt()
        self.tfidf = TfidfVectorizer(tokenizer=self.tokenizer)
        if cfg['idf_data_path'] is not None:
            self.init_tfidf(cfg['idf_data_path'])

    def init_tfidf(self, idf_data_path):
      dataset = nlp.data.TSVDataset(idf_data_path, field_indices=[0,1], num_discard_samples=1)

      self.tfidf.fit([i for i,_ in dataset])



    def tokenizer(self, raw_texts, pos=["Noun","Alpha","Number"], stopword=[]):
        p = self.okt.pos(raw_texts, 
            norm=True,   # 정규화(normalization)
            stem=True    # 어간추출(stemming)
            )
        o = [word for word, tag in p if len(word) > 1 and tag in pos and word not in stopword]
        return o

    def predict(self, x):
        """
        일기 내용을 바탕으로 키워드를 반환하는 함수
        
        Args:
            x (str) : Diary Content, 
            
        Returns:
            str : Diary Keyword
        """
        
        # 오타 수정(필요할 경우에)
        x = self.okt.normalize(x)
        x = [x]
        
        try:
            vectors = self.tfidf.fit_transform(x)
        except:
            return [None for i in range(self.max_keywords)]


        
        dict_of_tokens={i[1]:i[0] for i in self.tfidf.vocabulary_.items()}

        tfidf_vectors = []

        for row in vectors:
            tfidf_vectors.append({dict_of_tokens[column]:value for (column,value) in zip(row.indices,row.data)})

        sorted_tfidf_vectors = sorted(tfidf_vectors[0].items(), key = lambda item: item[1], reverse = True)
        
        keywords = []

        num_keywords = min(self.max_keywords, len(sorted_tfidf_vectors))

        for keyword, _ in sorted_tfidf_vectors[:num_keywords]:
            keywords.append(keyword)


        for _ in range(self.max_keywords - num_keywords):
            keywords.append(None)
        
    
    
        return keywords
    
    

class KoBERT(KeywordModel):
    def __init__(self, cfg):
        super().__init__(cfg)
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
        일기 내용을 바탕으로 키워드를 반환하는 함수
        
        Args:
            x (str) : Diary Content, 
            
        Returns:
            str : Diary Keyword
        """
    
    
        # 오타 수정(필요할 경우에)
        x = self.okt.normalize(x)
        
        tokenized_x = self.okt.pos(x)
        tokenized_nouns = ' '.join([word[0] for word in tokenized_x if word[1] == 'Noun' or word[1] == 'Number' or word[1] == 'Alpha'])
        
        if len(tokenized_nouns) == 0:
            return [None for i in range(self.max_keywords)]


        n_gram_range = (0, 1)
        try:
            count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
        except:
            return [None for i in range(self.max_keywords)]
        candidates = count.get_feature_names_out()
        
        x = split_sentences(x, num_workers=self.num_workers)
        
        x_datasets = KoBERTDataset(x, self.tok, self.max_len)
        x_dataloader = DataLoader(x_datasets, batch_size = self.batch_size, num_workers=self.num_workers)
        
        candidates_dataset = KoBERTDataset(candidates, self.tok, self.max_len)
        candidates_dataloader = DataLoader(candidates_dataset, batch_size = self.batch_size, num_workers=self.num_workers)
        
        x_all_embedding = torch.Tensor().to(self.device)
        candidates_all_embeddings = []
        with torch.no_grad():
            
            for token_ids, valid_length, segment_ids in x_dataloader:
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                
                x_embedding, _ = self.model(token_ids, valid_length, segment_ids)
                # x_embedding = x_embedding.to('cpu')
                # x_embedding = np.asarray([emb.numpy() for emb in x_embedding])
                
                x_all_embedding = torch.cat((x_all_embedding, x_embedding), dim=0)
                
            x_all_embedding = x_all_embedding.mean(dim=0).unsqueeze(dim=0)

            for token_ids, valid_length, segment_ids in candidates_dataloader:
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                
                candidates_embedding, _ = self.model(token_ids, valid_length, segment_ids)
                candidates_embedding = candidates_embedding.to('cpu')
                candidates_embedding = np.asarray([emb.numpy() for emb in candidates_embedding])
                
                candidates_all_embeddings.extend(candidates_embedding)

            


        x_all_embedding = np.asarray([emb.numpy() for emb in x_all_embedding.to('cpu')])
        distances = cosine_similarity(x_all_embedding, candidates_all_embeddings)


        num_keywords = min(self.max_keywords, distances.shape[-1])
        keywords = [candidates[index] for index in distances.argsort()[0][-num_keywords:]]

        for _ in range(self.max_keywords - num_keywords):
            keywords.append(None)


        return keywords


# Noe Use!
class SBERT(KeywordModel):
    """
    Args:
    """

    def __init__(self, cfg):
        """
        
        Args
        
        """
        self.model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens', device=cfg['device'])
        
        self.device = cfg['device']
        self.num_keywords = cfg['num_keywords']
        
        self.max_len = cfg['max_len']
        self.batch_size = cfg['batch_size']
        
        self.num_workers = cfg['num_workers']
        

        if cfg['model_path'] is not None:
            self.model.load_state_dict(torch.load(cfg['model_path']), map_location=self.device)
            
        self.model.eval()
        
        self.okt = Okt()


    def predict(self, x):
        """
        일기 내용을 바탕으로 키워드를 반환하는 함수
        
        Args:
            x (str) : Diary Content, 
            
        Returns:
            str : Diary Keyword
        """
        
        # 오타 수정(필요할 경우에)
        x = self.okt.normalize(x)
        
        tokenized_x = self.okt.pos(x)
        tokenized_nouns = ' '.join([word[0] for word in tokenized_x if word[1] == 'Noun' or word[1] == 'Number' or word[1] == 'Alpha'])
        # tokenized_nouns = self.tokenizer(x)
        # tokenized_nouns = ' '.join([word[0] for word in tokenized_nouns])
        
        n_gram_range = (0, 1)
        try:
            count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
        except:
            return [None for _ in range(self.max_keywords)]
        candidates = count.get_feature_names_out()


        doc_embedding = self.model.encode([x])
        candidate_embedding = self.model.encode(candidates)
        
        distances = cosine_similarity(doc_embedding, candidate_embedding)
        num_keywords = min(self.num_keywords, distances.shape[-1])
        keywords = [candidates[index] for index in distances.argsort()[0][-num_keywords:]]

        for _ in range(self.max_keywords - num_keywords):
            keywords.append(None)

        return keywords
