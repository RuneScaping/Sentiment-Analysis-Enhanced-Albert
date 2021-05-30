import torch
import torch.nn.functional as F

from transformers import (AlbertConfig,
                          AlbertForSequenceClassification, 
                          AlbertTokenizer,
                          )


class SentimentAnalyzer:
    def __init__(self, path='model', model_type='albert-base-v2'):
        self.path = path
        se