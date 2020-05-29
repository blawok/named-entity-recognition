import argparse, os
import numpy as np
import pandas as pd
import json
import subprocess
import sys

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Embedding,
    SpatialDropout1D,
    Bidirectional,
    LSTM,
    TimeDistributed,
    Dense
)


class BiLSTM(Model):

    def __init__(self):
        super(BiLSTM, self).__init__()
        self.embedding = Embedding(input_dim=N_WORDS, output_dim=100, input_length=MAX_LEN)
        self.spatial_dropout = SpatialDropout1D(0.1)
        self.bilstm = Bidirectional(LSTM(units=256, return_sequences=True))
        self.tddense = TimeDistributed(Dense(N_TAGS, activation="softmax"))

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.spatial_dropout(x)
        x = self.bilstm(x)
        return self.tddense(x)

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
if __name__ == '__main__':
    
#     install('tensorflow-addons==0.10.0')
#     from crf import CRF

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    
    parser.add_argument('--max-len', type=int, default=50)
    parser.add_argument('--n-tags', type=int, default=17)
    parser.add_argument('--n-words', type=int, default=35178)   
    
    parser.add_argument('--model-version', type=str, default='1')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    
    args, _ = parser.parse_known_args()
    
    MAX_LEN    = args.max_len
    N_TAGS     = args.n_tags
    N_WORDS    = args.n_words
    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch_size
    MODEL_VER  = args.model_version
    MODEL_DIR  = args.model_dir
    TRAINING_DIR  = args.training
    
    # ----- LOAD DATA -----
    print(TRAINING_DIR)
    train = pd.read_csv(os.path.join(TRAINING_DIR, 'bilstm_train.csv'),
                        header=None,
                        names=None
                       )

    train_y = train.iloc[:,:-50].values
    train_X = train.iloc[:,50:].values
    
    # ----- DECLARE MODEL -----
#     model = Sequential()
#     model.add(Embedding(input_dim=N_WORDS+1, output_dim=50, input_length=MAX_LEN, mask_zero=True))
#     model.add(SpatialDropout1D(0.1))
#     model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
#     model.add(Dense(N_TAGS))
#     crf = CRF(N_TAGS, sparse_target=True)
#     model.add(crf)
#     model.compile('adam', loss=crf.loss, metrics=[crf.accuracy])
    
    model = BiLSTM()
    model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy")
    
    # ----- FIT MODEL -----
    model.fit(train_X, 
#               np.array(train_y),
              train_y.reshape(*train_y.shape, 1),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.1,
              verbose=2)
    
    # ----- SAVE MODEL -----
    model.save(os.path.join(MODEL_DIR,'bi_lstm',MODEL_VER),
               save_format='tf')


