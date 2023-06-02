import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig


"""
Different types of NLP models for the harmonizer. 
LR and MLP are based on the pre-computed embedding for the input.
BERT models take the token ids as input. 
"""


# Base models built on precomputed embeddings
class LR(nn.Module):
    def __init__(self, input_size, n_class) -> None:
        super(LR, self).__init__() 
        if n_class <= 2:
            n_class = 1
        self.linear = nn.Linear(input_size, n_class)
        
    def forward(self, x):
        return self.linear(x)
    
    
class MLP(nn.Module):
    def __init__(self, input_size, n_class, hidden_size=100) -> None:
        super(MLP, self).__init__() 
        if n_class <= 2:
            n_class = 1
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_class))
        
    def forward(self, x):
        return self.mlp(x)
    
    

def get_token_ids(batch_text, bert_model):
    """
    Get token IDs for batch text input to ensure equal length.
    """
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    return tokenizer.batch_encode_plus(batch_text, padding='longest', truncation=True)['input_ids']


def get_entity_masks(batch_examples, bert_model):
    """
    Get token IDs and entity masks for batch examples of relation extraction.
    Each example contains the raw text input and the character spans for the two entities.
    Output: batch of token IDs and entity masks for the spans, all of same length.
    """
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    
    input_ids = []
    e1_mask = []
    e2_mask = []
    for ex in batch_examples:
        text = ex['text']
        s1 = ex['span1']
        s2 = ex['span2']
        ids, m1, m2 = tokenize_with_breakpoints(text, s1, s2,tokenizer)
        input_ids.append(ids)
        e1_mask.append(m1)
        e2_mask.append(m2)
    # pad to same length
    max_len = max([len(ids) for ids in input_ids])
    for i in range(len(input_ids)):
        l = len(input_ids[i])
        input_ids[i] += [0] * (max_len - l)
        e1_mask[i] += [0] * (max_len - l)
        e2_mask[i] += [0] * (max_len - l)
        
    return input_ids, e1_mask, e2_mask

    
    
        
        
def tokenize_with_breakpoints(text, span1, span2, tokenizer):
    '''
      Helper function to tokenize the text with the given spans as breakpoints
      Return the token IDs and span masks for the single input example.
    '''
    switched = False
    if span1[0] > span2[0]:
        # span position switched
        span1, span2 = span2[:], span1[:]
        switched = True
        
    tokens1 = [tokenizer.cls_token] + tokenizer.tokenize(text[:span1[0]])
    entity1 = tokenizer.tokenize(text[span1[0]:span1[1]])
    tokens2 = tokenizer.tokenize(text[span1[1]:span2[0]])
    entity2 = tokenizer.tokenize(text[span2[0]:span2[1]])
    tokens3 = tokenizer.tokenize(text[span2[1]:]) + [tokenizer.sep_token]
    
    if len(entity1)==0 or len(entity2)==0:
        print(text, span1, span2)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens1+entity1+tokens2+entity2+tokens3)
    mask1 = [0]*len(tokens1) + [1/len(entity1)]*len(entity1) + [0]*len(tokens2+entity2+tokens3)
    mask2 = [0]*len(tokens1+entity1+tokens2) + [1/len(entity2)]*len(entity2) + [0]*len(tokens3)
    
    # truncate the length
    input_ids = input_ids[:512]
    mask1 = mask1[:512]
    mask2 = mask2[:512]
    
    if not switched:
        return input_ids, mask1, mask2
    else:
        return input_ids, mask2, mask1
        
    
    

# BERT models from token IDs    
class BERTclf(nn.Module):
    """
    BERT model for text classification. The input should be token ids.
    """
    def __init__(self, n_class, bert_model) -> None:
        # bert_model is the model name string
        # for biomed tasks, use microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
        super(BERTclf, self).__init__() 
        
        if n_class <= 2:
            n_class = 1
            
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = AutoModel.from_pretrained(bert_model)
        self.config = AutoConfig.from_pretrained(bert_model)
        
        self.linear = nn.Linear(self.config.hidden_size, n_class)
        
    def forward(self, input_ids):
        return self.linear(self.get_embedding(input_ids))
    
    def get_embedding(self, input_ids):
        """
        Embedding before the last classification layer. 
        Use the embedding of the [CLS] token.
        Useful for precomputing features for LR and MLP.
        """
        return self.model(input_ids).last_hidden_state[:,0,:]
    
    def get_token_ids(self, text):
        input = self.tokenizer(text, truncation=True)
        return input['input_ids']
    
    
    
class BERTre(nn.Module):
    """
    BERT model for entity relation classification. The input should be token ids and entity masks.
    """
    def __init__(self, n_class, bert_model) -> None:
        # bert_model is the model name string
        # for biomed tasks, use microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
        super(BERTre, self).__init__() 
        
        if n_class <= 2:
            n_class = 1
            
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = AutoModel.from_pretrained(bert_model)
        self.config = AutoConfig.from_pretrained(bert_model)
        
        self.linear = nn.Linear(3*self.config.hidden_size, n_class)
        
        
    def forward(self, input_ids, e1_mask, e2_mask):
        return self.linear(self.get_embedding(input_ids, e1_mask, e2_mask))
    
    
    def get_embedding(self, input_ids, e1_mask, e2_mask):
        """
        Embedding before the last classification layer. 
        The relation extraction embedding is computed by concatenating
        the [CLS] token embedding, and the mean-pooled embeddings for the two entity spans.
        Useful for precomputing features for LR and MLP.
        """
        output = self.model(torch.as_tensor(input_ids)).last_hidden_state
        e0 = output[:,0,:]
        e1 = torch.bmm(e1_mask.unsqueeze(1).float(), output).squeeze(1)
        e2 = torch.bmm(e2_mask.unsqueeze(1).float(), output).squeeze(1)
        return torch.cat((e0, e1, e2), dim=-1)
        
    
    
