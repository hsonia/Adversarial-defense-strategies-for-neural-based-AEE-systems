from dataset_for_model import SentencesDataset
from model_generator import Generator

import numpy as np
import pickle
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Train:
  def __init__(self, dataset, model, batch_size = 32, Epochs = 50):
    
    self.dataset = dataset
    self.model = model

    self.batch_size = batch_size
    self.Epochs = Epochs

    self.df_loader = DataLoader(dataset, batch_size=self.batch_size, drop_last=True)
    
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    self.cnt = 0
    self.min_loss = np.inf

  def __call__(self):
    for epoch in range(self.Epochs):
      loss = (self.train(epoch))/len(self.dataset)

      if self.early_stop(epoch, loss, patience=10):
        break
  
  def train(self, epoch):
    prev = time.time()
    avg_loss = 0
    for i, sample in enumerate(self.df_loader):
      inp, target, attn_mask, mask = sample

      self.optimizer.zero_grad()
      pred_wrd = self.model(inp, mask, attn_mask)

      # msk = mask.unsqueeze(-1).expand_as(pred_wrd)
      # pred_wrd = pred_wrd.masked_fill(msk, 0)

      loss_pred_wrd = self.criterion(pred_wrd.transpose(1,2), target).to(device)
      
      elapsed = time.gmtime(time.time() - prev)
      print("Epoch:", epoch, ", batch:", i, ", time:", elapsed.tm_sec, "ms, Loss:", loss_pred_wrd)
      
      avg_loss += loss_pred_wrd
       

      loss_pred_wrd.backward()
      self.optimizer.step()
      
    #   target = target.masked_select(~mask)
    #   pred_wrd = pred_wrd.argmax(-1).masked_select(~mask)

    #   accuracy = float((target == pred_wrd).sum())
    #   accuracy = round(accuracy * 100 / target.size(0), 2)


           

    return avg_loss

  def early_stop(self, epoch, loss, patience=20):
      if loss < self.min_loss:
        self.min_loss = loss
        self.cnt = 0
        self._save_checkpoint(epoch, loss)
      elif self.min_loss <= loss+1e-5:
        self.cnt += 1
        if self.cnt == patience:
          return True
      return False

  def _save_checkpoint(self, epoch, loss):
    name = f'generator_{epoch}_{round(time.time(), 2)}.pt'
    
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'model_state': self.model.state_dict(),
        'optimizer_state': self.optimizer.state_dict()
    }, name)

    print()
    print("*** ", name, " saved. ***")
    
    
if __name__ == '__main__':
    
    with open('corpus_dict_glv.pkl', 'rb') as f:
        corpus_dictionary = pickle.load(f)

    with open('clean_sents.pkl', 'rb') as f:
        clean_sents = pickle.load(f)

    dataset = SentencesDataset(clean_sents, corpus_dictionary)

    embedding_dim = maxLen = (dataset.seqLen)
    vocab_size = len(dataset.corpus_dictionary) #vocab_size = 23793
    dropout = 0.05
    # nbr_layers = nbr_heads = 2
    nbr_layers , nbr_heads = 4, 4


    model = Generator(vocab_size, maxLen, embedding_dim, dropout, nbr_layers, nbr_heads).to(device)


    dataset_train, _ = train_test_split(dataset, test_size=0.2, random_state = 41)
    train = Train(dataset_train, model, batch_size = 32, Epochs = 20)

    train()
    
    joblib.dump(model, 'generator.joblib')
