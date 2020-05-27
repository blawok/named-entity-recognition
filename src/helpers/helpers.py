import numpy as np

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
        
        
def predictions_to_label(pred):
    tag2idx = {'I-geo': 0,
                'O': 1,
                'I-tim': 2,
                'I-gpe': 3,
                'I-eve': 4,
                'B-tim': 5,
                'I-nat': 6,
                'B-geo': 7,
                'B-eve': 8,
                'B-art': 9,
                'I-org': 10,
                'B-nat': 11,
                'I-art': 12,
                'I-per': 13,
                'B-org': 14,
                'B-per': 15,
                'B-gpe': 16}
    idx2tag = {i: w for w, i in tag2idx.items()}
    
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
        
    return out
