from krwordrank.word import KRWordRank
from krwordrank.hangle import normalize
from kss import split_sentences


class KeyWordExtractor:
    def __init__(self,min_count=5, max_length=10, keyword_number=3, max_iter=10, beta=0.85):
        self.wordrank_extractor = KRWordRank(
            min_count=min_count,
            max_length=max_length,
            verbose=True
        )
        
        self.beta = beta
        self.max_iter = max_iter
        self.keyword_number = keyword_number
    
    def extraction(self, contents):
        contents = normalize(contents, english=True, number=True)
        contents = split_sentences(contents, num_workers=1)
        
        keywords, _, _ = self.wordrank_extractor.extract(contents, self.beta, self.max_iter,self.keyword_number)
       
        return list(keywords.keys())

    

        
    