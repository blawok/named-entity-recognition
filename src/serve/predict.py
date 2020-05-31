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
from transformers import RobertaForTokenClassification

def model_fn(model_dir):
    print("Loading model.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RobertaForTokenClassification.from_pretrained(model_dir,
                                                          num_labels=20)
    
    model.to(device).eval()

    print("Done loading model.")
    return model


def predict_fn(input_data, model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_tags = input_data[input_data.columns[0:45]].values
    val_inputs = input_data[input_data.columns[45:90]].values
    val_masks = input_data[input_data.columns[90:135]].values
    val_segs = input_data[input_data.columns[135:]].values

    val_inputs = torch.tensor(val_inputs)
    val_tags = torch.tensor(val_tags)
    val_masks = torch.tensor(val_masks)
    val_segs = torch.tensor(val_segs)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=32)
    
    tag2name = {14: 'B-art',
                16: 'B-eve',
                0: 'B-geo',
                13: 'B-gpe',
                12: 'B-nat',
                10: 'B-org',
                4: 'B-per',
                2: 'B-tim',
                5: 'I-art',
                7: 'I-eve',
                15: 'I-geo',
                8: 'I-gpe',
                11: 'I-nat',
                3: 'I-org',
                6: 'I-per',
                1: 'I-tim',
                17: 'X',
                9: 'O',
                18: '[CLS]',
                19: '[SEP]'}   
    
    # ---- START EVALUATION ----
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []

    print("***** Running evaluation *****")
    print("  Num examples ={}".format(len(val_inputs)))
    print("  Batch size = {}".format(batch_num))
    
    for step, batch in enumerate(valid_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch

        with torch.no_grad():
            outputs = predictor(input_ids, token_type_ids=None, attention_mask=input_mask,)
            logits = outputs[0] 

        # Get NER predict result
        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        logits = logits.detach().cpu().numpy()

        # Get NER true result
        label_ids = label_ids.to('cpu').numpy()

        # Only predict the real word, mark=0, will not calculate
        input_mask = input_mask.to('cpu').numpy()

        # Compare the valuable predict result
        for i,mask in enumerate(input_mask):
            temp_1 = []
            temp_2 = []

            for j, m in enumerate(mask):
                if m:
                    if tag2name[label_ids[i][j]] != "X" and tag2name[label_ids[i][j]] != "[CLS]" and tag2name[label_ids[i][j]] != "[SEP]" : # Exclude the X label
                        temp_1.append(tag2name[label_ids[i][j]])
                        temp_2.append(tag2name[logits[i][j]])
                else:
                    break

            y_true.append(temp_1)
            y_pred.append(temp_2)
            
    return [y_true, y_pred]