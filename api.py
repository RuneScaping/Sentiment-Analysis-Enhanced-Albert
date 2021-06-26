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
        self.model = AlbertForSequenceClassification.from_pretrained(self.path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def convert_to_features(self, sentence):
        
        text_a = sentence
        text_b = None
        max_length = 512
        pad_on_left = False
        pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        pad_token_segment_id = 0
        mask_padding_with_zero = True
            
        inputs = self.tokenizer.encode_plus(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=max_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attentio