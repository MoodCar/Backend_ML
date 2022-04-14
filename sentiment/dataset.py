import torch
import torch.nn as nn
import gluonnlp as nlp
import numpy as np


class BERTDataset(torch.utils.Dataset):
    def __init__(self, dataset, bert_tokenizer, max_len,
                 pad=True, pair=False):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i]) for i in dataset]
        # self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i],)

    def __len__(self):
        return (len(self.sentences))
