import argparse
import json
import os
import pickle
import subprocess
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import (TensorDataset,
                              DataLoader,
                              RandomSampler,
                              SequentialSampler)
from utils import install, _get_tag2name, _get_data_loader


def model_fn(model_dir):
    """
    Method for loading model artifacts and creating model from them.
    model_dir    - model artifacts directory in S3
    """
    print("Loading model.")
    
    from transformers import RobertaForTokenClassification
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RobertaForTokenClassification.from_pretrained('roberta-base',
                                                          num_labels=20)
        
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
            
    return model.to(device)


def train(model, train_loader, epochs, optimizer, device):
    """
    Training method that is called by the PyTorch training script.
    The parameters passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    print('--------------------')
    print('--------------------')
    print('Starting training.')
    
    for _ in trange(epochs, desc="Epoch"):
        model.train()
        training_loss = 0
        training_steps = 0
        
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            batch_input_ids, batch_input_mask, batch_labels = batch

            # forward pass
            outputs = model(batch_input_ids,
                            token_type_ids=None,
                            attention_mask=batch_input_mask,
                            labels=batch_labels
                           )
            loss, scores = outputs[:2]

            # backward pass
            loss.backward()

            # cummulate training loss
            training_loss += loss.item()
            training_steps += 1

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=1.0)

            # update parameters
            optimizer.step()
            optimizer.zero_grad()

        print("Train loss: {}".format(training_loss/training_steps))
    
    
def evaluate(model, test_loader, device):
    """
    Method for RoBERTa evaluation, requires ready to go DataLoader with test set.
    The parameters passed are as follows:
    model        - The PyTorch model that we wish to train.
    test_loader  - The PyTorch DataLoader that should be used during evaluation.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    print('--------------------')
    print('--------------------')
    print('Evaluation.')
    model.eval()
    
    y_true = []
    y_pred = []
    
    tag2name = _get_tag2name()
    
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch

        with torch.no_grad():
            outputs = model(input_ids,
                            token_type_ids=None,
                            attention_mask=input_mask,)
            logits = outputs[0] 

        # extract predictions
        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        logits = logits.detach().cpu().numpy()

        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        # omit additional tokens
        for i,mask in enumerate(input_mask):
            temp_y_true = []
            temp_y_pred = []
            for j, m in enumerate(mask):
                if m:
                    if (tag2name[label_ids[i][j]] != "X" and
                        tag2name[label_ids[i][j]] != "[CLS]" and
                        tag2name[label_ids[i][j]] != "[SEP]"):
                        temp_y_true.append(tag2name[label_ids[i][j]])
                        temp_y_pred.append(tag2name[logits[i][j]])
                else:
                    break

            y_true.append(temp_y_true)
            y_pred.append(temp_y_pred)

    print("Test Set F1 Score: %f"%(f1_score(y_true, y_pred)))
    print('--------------------')
    
    
if __name__ == '__main__':
    
    install('transformers')
    install('tqdm')
    install('seqeval')
    
    import subprocess
    import sys
    from transformers import RobertaConfig, RobertaForTokenClassification, AdamW
    from tqdm import tqdm,trange
    from seqeval.metrics import f1_score
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--max_len', type=int, default=45)   
    parser.add_argument('--n_tags', type=int, default=20)
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--val-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # ------ LOAD DATA ------
    train_loader = _get_data_loader(args.batch_size, args.data_dir, "train_roberta.csv", args.max_len)
    test_loader = _get_data_loader(args.batch_size, args.val_dir, "test_roberta.csv", args.max_len)
    
    # ------ CREATE ROBERTA MODEL ------
    model = RobertaForTokenClassification.from_pretrained('roberta-base',
                                                          num_labels=args.n_tags).to(device)

    # ------ SPECIFY OPTIMIZER ------
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)

    # ------ TRAIN ROBERTA ------
    train(model, train_loader, args.epochs, optimizer, device)
    
    # ------ EVALUATE ROBERTA ------
    evaluate(model, test_loader, device)
    
    # ------ SAVE ROBERTA FILES ------
    print("Saving the model.")
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)
    
    