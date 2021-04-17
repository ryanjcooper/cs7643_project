from interface import implements, Interface
from .LanguageModelTranslator import LanguageModelTranslator
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

class Word2Vec(implements(LanguageModelTranslator)):
    def __init__(self, path='../data/glove.6B/glove.6B.200d.txt'):
        self.model = None
        self.path = path

        self.load_model()
        pass
    
    def convert(self, words_list, method='average'):
        vec = []
        for word in words_list:
            try:
                vec.append(self.model[word.lower()])
            except:
                continue
        
        if method == 'average':
            vec = np.array(vec).mean(axis=0)
            
        else:
            raise NotImplementedError
            
        return vec
    
    def load_model(self):
        self.model = {}
        f = open(self.path, encoding='utf8')
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32') 
            self.model[word] = coefs
        f.close()
        
        