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
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")
    
    from transformers import RobertaForTokenClassification
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RobertaForTokenClassification.from_pretrained(model_dir,
                                                          num_labels=20)
    
    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_input = pd.read_csv(os.path.join(training_dir, "train_roberta.csv"), header=None, names=None)
    
    tr_tags = train_input[train_input.columns[0:45]].values
    tr_inputs = train_input[train_input.columns[45:90]].values
    tr_masks = train_input[train_input.columns[90:135]].values
    tr_segs = train_input[train_input.columns[135:]].values
    
    tr_inputs = torch.tensor(tr_inputs)
    tr_tags = torch.tensor(tr_tags)
    tr_masks = torch.tensor(tr_masks)
    tr_segs = torch.tensor(tr_segs)
    
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=batch_size,
                                  drop_last=True)
    

    return train_dataloader


def train(model, train_loader, epochs, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    print('Starting training')
    for _ in trange(epochs, desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_loader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # forward pass
            outputs = model(b_input_ids, token_type_ids=None,
            attention_mask=b_input_mask, labels=b_labels)
            loss, scores = outputs[:2]

            # backward pass
            loss.backward()

            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)

            # update parameters
            optimizer.step()
            optimizer.zero_grad()

        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        


if __name__ == '__main__':
    
    install('transformers')
    from transformers import RobertaConfig, RobertaForTokenClassification, AdamW
    install('tqdm')
    from tqdm import tqdm,trange
    
    parser = argparse.ArgumentParser()

    # SageMaker Parameters
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # ------ CREATE ROBERTA MODEL ------
    model = RobertaForTokenClassification.from_pretrained('roberta-base',
                                                          num_labels=20).to(device)

    # ------ TRAIN ROBERTA ------
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

    train(model, train_loader, args.epochs, optimizer, device)

    # ------ SAVE ROBERTA FILES ------
    bert_out_address = args.model_dir
    output_model_file = os.path.join(bert_out_address, "pytorch_model.bin")
    output_config_file = os.path.join(bert_out_address, "config.json")
    
    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)
#     tokenizer.save_vocabulary(bert_out_address)
    
    