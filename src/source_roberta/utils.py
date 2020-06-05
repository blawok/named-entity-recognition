import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def install(package):
    """
    Method for installing a package
    package    - package name
    """
    subprocess.call([sys.executable, "-m", "pip", "install", package])

def _get_data_loader(batch_size, data_s3_dir, data_file_name, max_len):
    """
    Method for loading data and transforming it for traning and inference.
    data_s3_dir    - directory of file to load from S3
    data_file_name - name of a file to be loaded
    """
    print("Get data loader.")

    input_data = pd.read_csv(os.path.join(data_s3_dir, data_file_name),
                             header=None,
                             names=None)
    
    tags = input_data[input_data.columns[0:max_len]].values
    inputs = input_data[input_data.columns[max_len:(2*max_len)]].values
    masks = input_data[input_data.columns[(2*max_len):(3*max_len)]].values
    segs = input_data[input_data.columns[(3*max_len):]].values
    
    inputs = torch.tensor(inputs)
    tags = torch.tensor(tags)
    masks = torch.tensor(masks)
    segs = torch.tensor(segs)
    
    data = TensorDataset(inputs, masks, tags)
    sampler = RandomSampler(data)
    
    dataloader = DataLoader(data,
                            sampler=sampler,
                            batch_size=batch_size,
                            drop_last=True)

    return dataloader

def _get_tag2name():
    return {14: 'B-art',
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
            19: '[SEP]'
           }  