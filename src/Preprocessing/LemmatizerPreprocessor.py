from .Preprocessor import Preprocessor
from interface import implements, Interface

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

# download nltk datasets
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class LemmatizerPreprocessor(implements(Preprocessor)):
    def __init__(self,):
        self.lemmatizer = WordNetLemmatizer() 
        self.stop_words = set(stopwords.words('english')) # set stopwords to english 

    def clean(self, s):
        txt = s.lower()
        txt = re.sub(r'[^a-z ]', ' ', txt)
        tokenized_words=word_tokenize(txt)

        filtered_words = [w for w in tokenized_words if not w in self.stop_words]

        lemmitized_words = [self.lemmatizer.lemmatize(w) for w in filtered_words]

        return lemmitized_words

