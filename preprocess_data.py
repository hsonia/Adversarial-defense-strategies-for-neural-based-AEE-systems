import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
import pickle

class Preprocess:
    def __init__(self, data, X=None, y=None):
        
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        
        #importing dataset
        if X == None:
            self.dataset = data()
            self.X = self.dataset['essay'].tolist()
            self.y = self.dataset['domain1_score']
        else:
            self.X = X.tolist()
            self.y = y

        self.y.reset_index(drop = True, inplace = True)

    def __call__(self):
        clean_sents = [self.sent2word(sent) for sent in self.X]
        return clean_sents
    
    def sent2word(self, sent):
        sent = re.sub("[^a-zA-Z]", " ", sent)
        sent = sent.lower().split()
        words = [word for word in sent if word not in self.stop_words]
        return words
    
    #for glove, word2ev and tf-idf vector representation models
    def essays2sents2word(self):
        Sents = []
        for essay in self.X:
            Sents += self._essays2sents2word(essay)
        return Sents

    def _essays2sents2word(self, essay):
        raw_sentences = self.tokenizer.tokenize(essay.strip())
        sentences = [self.sent2word(sent) for sent in raw_sentences if len(sent)>0]
        return sentences
    

import subprocess
import sys

class VectorRepresentationModels:
    def __init__(self, preprocess):
        
        self.sents = preprocess.essays2sents2word()
        self.clean_sents = preprocess()
        
        
    def glove(self, epochs=500):
        #installing and importing required packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'glove-python-binary'])
        from glove import Corpus, Glove
        
        corpus = Corpus()
        #Training the corpus to generate the co-occurrence matrix which is used in GloVe
        corpus.fit(self.sents, window=50)
        
        model = Glove(no_components=50, learning_rate=0.09, alpha=0.75, max_count=len(corpus.dictionary), max_loss=10.0)
        model.fit(corpus.matrix, epochs=epochs, no_threads=4, verbose=True)
        model.add_dictionary(corpus.dictionary)
        
        with open('corpus_dict_glv.pkl', 'wb') as f:
            pickle.dump(corpus.dictionary, f)
        
        Dict = {}
        for word in corpus.dictionary:
            Dict[word] = model.word_vectors[corpus.dictionary[word]]
        
        return model, Dict, corpus
    
    def word2vec(self, num_features=300, min_word_count=40, context=10, downsampling=1e-3):
        #installing and importing required packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'gensim'])
        from gensim.models import Word2Vec
        
        model = Word2Vec(self.sents, size=num_features, window=context, min_count=min_word_count, workers=-1)
        model.init_sims(replace=True)
        model.wv.save_word2vec_format('word2VecModel.bin', binary=True)

        Dict = {}
        for word in model.wv.vocab:
            Dict[word] = model.wv[word]
            
        return model, Dict

    def tf_idf(self, stop_words='english', max_df=0.95, min_df=2, max_features=100000):
        
        #sents for tfidf shall be joined
        Sentidf = []
        for sent in self.sents:
          Sentidf.append(" ".join(sent))
        
        #importing required packages
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=max_df, min_df=min_df, max_features=max_features)
        bows = tfidf_vectorizer.fit_transform(Sentidf)
        corpus = tfidf_vectorizer.get_feature_names()
        return tfidf_vectorizer, bows
    
    def makeVector(self, words, model, num_features, Dict):
        vector = np.zeros((num_features,), dtype="float32")
        num_words = 0
        for word in words:
            #if word in model.dictionary:
            if word in Dict:
                num_words += 1
                vector = np.add(vector, Dict[word])
        if model == "word2vec":
            vector = np.divide(vector, num_words)
        return vector
    
    def getVector(self, model, num_features, Dict, samples=None):
        if samples == None:
          samples = self.clean_sents
        count = 0
        vectors = np.zeros((len(samples), num_features), dtype="float32")
        for essay in samples:
            vectors[count] = self.makeVector(essay, model, num_features, Dict)
            count += 1
        return vectors
    
    def cleanSent_vec(self, model, Dict, samples=None):

        inp=50 if "glove" in str(model) else 300

        vec = self.getVector(model, inp, Dict, samples)
        vec = np.array(vec)
        vec = np.reshape(vec, (vec.shape[0], 1, vec.shape[1]))

        return vec
        
        
        
if __name__ == '__main__':
    preprocess = Preprocess(Data())
    vrm = VectorRepresentationModels(preprocess)
    
    #create models and dicts
    glv_model, glv_Dict, corpus = vrm.glove(epochs=500)
    w2v_model, w2v_Dict = vrm.word2vec()
    tfidf_vectorizer, bows = vrm.tf_idf()
    
    #create vectors for ML models
    clean_sents = preprocess()
    glv_vec = vrm.cleanSent_vec(glv_model, glv_Dict)
    w2v_vec = vrm.cleanSent_vec(w2v_model, w2v_Dict)
    
    
    
    
            
        
        
        
        
        
        
