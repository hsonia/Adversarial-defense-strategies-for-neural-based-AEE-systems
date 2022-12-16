import pickle
import pandas as pd
import numpy as np
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentencesDataset(Dataset):

  def __init__(self, clean_sents, corpus_dictionary):

    self.clean_sents = clean_sents
    
    self.corpus_dictionary = self.update_corpus_dict(corpus_dictionary)
    
    self.maxLen = self.get_maxLen()
    
    self.seqLen = self.maxLen
    
    self.dataset = self.get_dataset()
  
  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    
    item = self.dataset.iloc[index]

    #model needs as inp: original_masked_idx, segments, attn_mask, original_idx
    inp = torch.Tensor(item["original_masked_idx"]).long()
    mask = torch.tensor([item["masked_idx"]], dtype=torch.bool).squeeze(0)
    target = torch.Tensor(item["original_idx"]).long()
    target = target.masked_fill_(mask, 0)

    attn_mask = (inp == self.corpus_dictionary['PAD']).unsqueeze(0)

    return(
        inp.to(device),
        target.to(device),
        attn_mask.to(device),
        mask.to(device)
    )
    
  
  def get_maxLen(self):
  
    #maximum sentence length
    maxLen = max([len(sent) for sent in self.clean_sents])
    return maxLen


  def get_dataset(self):


    dataset = self.__preprocess_sentences()

    return dataset
  
  def update_corpus_dict(self, corpus_dictionary):
    
    for key in corpus_dictionary:
      corpus_dictionary[key] += 4

    #token PAD to padd sequences to max_len
    corpus_dictionary['PAD'] = 0

    #token CLS indicates the beg of the sentence
    corpus_dictionary['CLS'] = 1

    #token MASK to mask words from the sentence
    corpus_dictionary['MASK'] = 2

    #token END indicates end of sentence
    corpus_dictionary['END'] = 3

    return corpus_dictionary

  def __preprocess_sentences(self):

    #creating index vectors for the sentences add appending 'CLS' at the beg of the sents
    original_idx  = [[self.corpus_dictionary['CLS']]+list(self.corpus_dictionary[word] for word in sent)
    +[self.corpus_dictionary['END']] for sent in self.clean_sents]


    #creating sentences with masked words and boolean inverse masked vector
    original_masked_idx, masked_idx = self.__mask_words(original_idx)

    dataset = pd.DataFrame()

    ##adjusting the length to maximum_len
    original_idx = pad_sequences(original_idx, padding="post", value=self.corpus_dictionary['PAD'])
    original_masked_idx = pad_sequences(original_masked_idx, padding="post", value=self.corpus_dictionary['PAD'])
    masked_idx = pad_sequences(masked_idx, dtype='object', padding="post", value=True)

    dataset['original_idx'] = [org_idx for org_idx in original_idx]
    dataset['original_masked_idx'] = [org_msk_idx for org_msk_idx in original_masked_idx]
    dataset['masked_idx'] = [msk_idx for msk_idx in masked_idx]
    
    self.seqLen = self.__update_seqLen(dataset)

    return dataset
  
  def __update_seqLen(self, dataset):
    return len(dataset['original_idx'].iloc[0])

  def __mask_words(self, original_idx):
    
    #creating masked index vectors for the sentences
    original_masked_idx  = [org_idx.copy() for org_idx in original_idx]

    #creating mask vectors for the sentences
    masked_idx = [list(True for _ in sent) for sent in original_masked_idx]

    #mask some words for prediction of all sentences
    for sent_idx in range(len(original_masked_idx)):

      # mask_word_idx = mask_percent * len(sent) (mask_percent for BERT = 15%)
      mask_word_ratio = round(random.uniform(0.15, 0.5) * len(original_masked_idx[sent_idx]))
      mask_word_idx = random.sample(range(len(original_masked_idx[sent_idx])), mask_word_ratio)

      for idx in mask_word_idx:
        masked_idx[sent_idx][idx] = False
        if random.random()<0.85:
          original_masked_idx[sent_idx][idx] = self.corpus_dictionary['MASK']
        else:
          original_masked_idx[sent_idx][idx] = random.sample(list(self.corpus_dictionary.values()), 1)[0]
    
    return original_masked_idx, masked_idx
