import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaForTokenClassification,RobertaTokenizer,RobertaConfig
from scipy.special import softmax
from tensorflow.keras.preprocessing.sequence import pad_sequences

def model_fn(model_dir):
    print("Loading model.")
    
    from transformers import RobertaForTokenClassification
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RobertaForTokenClassification.from_pretrained('roberta-base',
                                                          num_labels=20)
        
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
            
    return model.to(device)

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    print('Making predictions.')
    
    # ----- PREPROCESSING -----
    tokenizer=RobertaTokenizer.from_pretrained('roberta-base',do_lower_case=False)
    max_len  = 45
    
    tokenized_texts = []
    temp_token = []
    temp_token.append('[CLS]')
    token_list = tokenizer.tokenize(input_data)
    
    for m,token in enumerate(token_list):
        temp_token.append(token)
    if len(temp_token) > max_len-1:
        temp_token= temp_token[:max_len-1]
        
    temp_token.append('[SEP]')
    tokenized_texts.append(temp_token)
    
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_len, dtype="long", truncating="post", padding="post")
    
    attention_masks = [[int(i>0) for i in ii] for ii in input_ids]
    segment_ids = [[0] * len(input_id) for input_id in input_ids]
    
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    segment_ids = torch.tensor(segment_ids)
    
    # ----- MAKING PREDICTIONS -----
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None,attention_mask=None,)
        logits = outputs[0]    
    
    predict_results = logits.detach().cpu().numpy()
    result_arrays_soft = softmax(predict_results[0])
    
    result_list = np.argmax(result_array,axis=-1)
    
    return result_list