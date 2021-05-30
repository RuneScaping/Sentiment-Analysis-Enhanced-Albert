import torch
import torch.nn.functional as F

from transformers import (AlbertConfig,
                          AlbertForSequenceClassification, 
                          AlbertTokenizer,
                          )


class SentimentAnalyzer:
    def __init__(self, path='model', model_type='albert-base-v2'):
        self.path = path
        self.model_type = model_type
        self.tokenizer = AlbertTokenizer.from_pretrained(self.model_type, do_lower_case=True)
        self.mode