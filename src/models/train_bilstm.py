import argparse, os
import numpy as np
import pandas as pd
import json

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, SpatialDropout1D, Bidirectional


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    
    parser.add_argument('--max-len', type=int, default=50)
    parser.add_argument('--n-tags', type=int, default=17)
    parser.add_argument('--n-words', type=int, default=35178)   
    
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    
    args, _ = parser.parse_known_args()
    
    max_len    = args.max_len
    n_tags     = args.n_tags
    n_words    = args.n_words
    epochs     = args.epochs
    batch_size = args.batch_size
    model_dir  = args.model_dir
    training_dir   = args.training
    
    # ----- LOAD DATA -----
    train_sample = pd.read_csv(os.path.join(training_dir, 'train.csv'),
                               header=None,
                               names=None,
                               nrows=1024
                              )

    train_sample_y = train_sample.iloc[:,:-50].values
    train_sample_X = train_sample.iloc[:,50:].values
    
    # ----- DECLARE MODEL -----
    model = Sequential([

        Embedding(input_dim=n_words, output_dim=50, input_length=max_len),
        SpatialDropout1D(0.1),

        Bidirectional(LSTM(units=100, return_sequences=True)),
        TimeDistributed(Dense(n_tags, activation="softmax"))
    ])
    
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy")
    
    model.fit(train_sample_X, 
              train_sample_y.reshape(*train_sample_y.shape, 1),
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              verbose=2)
    
    model.save(os.path.join(model_dir,'bi_lstm/1'), save_format='tf')


