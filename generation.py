# -*- coding: utf-8 -*-
"""generation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m9VjCuInn3bs48MUM9BvBVt_rGE8OT4q
"""

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", 'nlpaug'])
import nlpaug.augmenter.word as naw
import random
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from sklearn.metrics import cohen_kappa_score, pairwise
import more_itertools

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class GEN_SAMPLES:
  def __init__(self, vrm, model, dictionary, syn, ant):
    self.model = model
    self.dictionary = dictionary
    self.vrm = vrm
    self.clean_sents = self.vrm.clean_sents
    self.syn = syn
    self.ant = ant
  
  def generate_samples(self):
    self.gen_samples, self.y_gen_grades = self.gen_sents()
    return self.gen_samples, self.y_gen_grades

  def gen_sents(self):
    self.gen_samples = []
    self.y_to_gen = []
    self.gen_pos()
    self.gen_neg()
    return self.gen_samples, self.y_to_gen

  
  def gen_pos(self):
    aug_syn = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', aug_min=1, aug_max=10, aug_p=0.3, lang='eng', 
                     stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, force_reload=False, 
                     verbose=0)
    num_pos_sents = random.randint(len(self.clean_sents)//2, len(self.clean_sents))
    pos_sents = list(set([random.randrange(0, len(self.clean_sents), 1) for i in range(num_pos_sents)]))
    


    for sent_index in pos_sents:
      sent = self.clean_sents[sent_index].copy()
      l = [aug_syn.augment(" ".join(sent)), [" ".join(self.pos_neg_sent(sent, "pos"))]]
      rnd = random.choices(l, k=random.randint(0, len(l)))
      aug = list(set(more_itertools.flatten(rnd)))
      for g in aug:

        self.y_to_gen.append((len(self.gen_samples), sent_index))
        self.gen_samples.append(g)

  
  def gen_neg(self):
    aug_ant = naw.AntonymAug(name='Antonym_Aug', aug_min=1, aug_max=10, aug_p=0.3, lang='eng', stopwords=None, tokenizer=None, 
                     reverse_tokenizer=None, stopwords_regex=None, verbose=0)
    aug_ran = naw.RandomWordAug(action='delete', name='RandomWord_Aug', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None, 
                        target_words=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, verbose=0)

    num_neg_sents = random.randint(len(self.clean_sents)//2, len(self.clean_sents))
    neg_sents = list(set([random.randrange(0, len(self.clean_sents), 1) for i in range(num_neg_sents)]))
    


    for sent_index in neg_sents:
      sent = self.clean_sents[sent_index].copy()
      l = [aug_ant.augment(" ".join(sent)), aug_ran.augment(" ".join(sent)), [" ".join(self.pos_neg_sent(sent, "neg"))]]
      rnd = random.choices(l, k=random.randint(0, len(l)))
      aug = list(set(more_itertools.flatten(rnd)))
      for g in aug:

        self.y_to_gen.append((len(self.gen_samples), sent_index))
        self.gen_samples.append(g)
    

      
  
  def pos_neg_sent(self, sent, mode):

    len_l = random.randint(len(sent)//2, len(sent)-1)
    l = list(set([random.randrange(0, len(sent)-1, 1) for _ in range(len_l)]))
    for word_index in l:
      word = sent[word_index]
      if mode == "pos":
        if word in self.syn.keys():
          # similar = model.most_similar(sent[word_index], 2)[0][0]
          sent[word_index] = self.syn[word]
      else:
        if word in self.syn.keys():
          sent[word_index] = self.ant[word]
          # similar = model.most_similar(sent[word_index], -1)[-1][0]
    return sent

class GEN_VECTORS:
  def __init__(self, samples, y_to_gen, vrm, model, dictionary, nn=False):
    self.samples = samples
    self.y_to_gen = y_to_gen
    self.vrm = vrm
    self.model = model
    self.dictionary = dictionary
    self.nn = nn

  
  def generate_vectors(self, y):
    if self.nn == False:
      samples = []
      self.y_to_gen_local = []
      for index_gen, index_org in self.y_to_gen:
        if index_org in y.index:
          self.y_to_gen_local.append((len(samples), index_org))
          samples.append(self.samples[index_gen])
      return self.vrm.cleanSent_vec(self.model, self.dictionary, samples)
    else:
      return self.vrm.cleanSent_vec(self.model, self.dictionary, self.samples)

  
  def cosine_similarity(self, a,b):
    return (1-pairwise.cosine_similarity(a,b))/2

  def generate_grades(self, gen_vec, vec, y):
    d = pd.DataFrame()
    d['vec'] = vec.copy().tolist()
    d.index = y.index
    y_gen = []

    if self.nn == True: y_to_gen = self.y_to_gen
    else: y_to_gen = self.y_to_gen_local
    
    for index_gen, index_org in y_to_gen:
      if index_org in y.index:
        per = self.cosine_similarity(gen_vec[index_gen], d['vec'][index_org])
        new_grade = y[index_org]*per.item()
        y_gen.append(np.around(new_grade))
    y_gen = pd.Series(y_gen)
    y_gen = y_gen.astype('int')
    
    return y_gen

class AUGSAM:
  def __init__(self, org_vec, aug_y, gen_vec, gen_y):
    self.org_vec = org_vec.copy().tolist()
    self.aug_y = aug_y
    self.gen_vec = gen_vec.copy().tolist()
    self.gen_y = gen_y
  
  def __call__(self):
    aug_vec, aug_y = self.augment_samples()
    return aug_vec, aug_y
  
  def augment_samples(self):
    aug_vec = self.org_vec
    aug_vec.extend(self.gen_vec)
    aug_vec = np.array(aug_vec)

    self.aug_y = self.aug_y.append(self.gen_y, ignore_index = True)

    return aug_vec, self.aug_y

if __name__ == '__main__':

  from data import Data
  from preprocess_data import Preprocess, VectorRepresentationModels

  data = Data()
  preprocess = Preprocess(data)
  vrm = VectorRepresentationModels(preprocess)
  
  #create models and dicts
  glv_model, glv_Dict, corpus = vrm.glove(epochs=500)
  w2v_model, w2v_Dict = vrm.word2vec()
  tfidf_vectorizer, bows = vrm.tf_idf()
  
  #create vectors for ML models
  clean_sents = preprocess()
  glv_vec = vrm.cleanSent_vec(glv_model, glv_Dict)
  w2v_vec = vrm.cleanSent_vec(w2v_model, w2v_Dict)

  glv_syn = {}
  for word in glv_model.dictionary:
    glv_syn[word] = glv_model.most_similar(word, 2)[0][0]
    
  glv_ant = {}
  for word in glv_model.dictionary:
    glv_ant[word] = glv_model.most_similar(word, -1)[-1][0]

  w2v_syn = {}
  for word in w2v_Dict.keys():
    w2v_syn[word] = w2v_model.wv.most_similar(word)[0][0]

  w2v_ant = {}
  for word in w2v_Dict.keys():
    w2v_ant[word] = w2v_model.wv.most_similar(negative=[word])[0][0]
  

  y = preprocess.y
  from sklearn.model_selection import train_test_split
  import pickle
  
  #generate new data using glove model
  X_train_glv, X_test_glv, y_train_glv, y_test_glv = train_test_split(glv_vec, y, test_size=0.25, random_state=41)
  gen_samples_glv = GEN_SAMPLES(vrm, glv_model, glv_Dict, glv_syn, glv_ant)
  new_samples_glv, new_grad_glv = gen_samples_glv.generate_samples()
  
  #generate new data using w2v model
  X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(w2v_vec, y, test_size=0.25, random_state=41)
  gen_samples_w2v = GEN_SAMPLES(vrm, w2v_model, w2v_Dict, w2v_syn, w2v_ant)
  new_samples_w2v, new_grad_w2v = gen_samples_w2v.generate_samples()

  #creation of new vectors using glove model
  gen_vec_glv = GEN_VECTORS(new_samples_glv, new_grad_glv, vrm, glv_model, glv_Dict)
  new_x_train_glv = gen_vec_glv.generate_vectors(y_train_glv)
  new_y_train_glv = gen_vec_glv.generate_grades(new_x_train_glv, X_train_glv, y_train_glv)

  #creation of new vectors using w2v model
  gen_vec_w2v = GEN_VECTORS(new_samples_w2v, new_grad_w2v, vrm, w2v_model, w2v_Dict)
  new_x_train_w2v = gen_vec_w2v.generate_vectors(y_train_w2v)
  new_y_train_w2v = gen_vec_w2v.generate_grades(new_x_train_w2v, X_train_w2v, y_train_w2v)

  #data augmentation
  aug_vec_glv, aug_y_glv = AUGSAM(X_train_glv, y_train_glv, new_x_train_glv, new_y_train_glv)()
  aug_vec_w2v, aug_y_w2v = AUGSAM(X_train_w2v, y_train_w2v, new_x_train_w2v, new_y_train_w2v)()


  #saving files
  file = open('org_gen_aug_glv.pkl','wb')
  pickle.dump(X_train_glv, file)
  pickle.dump(y_train_glv, file)
  pickle.dump(new_x_train_glv, file)
  pickle.dump(new_y_train_glv, file)
  pickle.dump(aug_vec_glv, file)
  pickle.dump(aug_y_glv, file)

  file1 = open('org_gen_aug_w2v.pkl','wb')
  pickle.dump(X_train_w2v, file1)
  pickle.dump(y_train_w2v, file1)
  pickle.dump(new_x_train_w2v, file1)
  pickle.dump(new_y_train_w2v, file1)
  pickle.dump(aug_vec_w2v, file1)
  pickle.dump(aug_y_w2v, file1)