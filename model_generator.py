import torch
from torch import nn
import random
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomEmbedding(nn.Module):
  def __init__(self, vocab_size, embedding_dim, maxLen):
    super(CustomEmbedding, self).__init__()
    

    #variable initialization
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.maxLen = maxLen

    #layers initialization
    self.inp_Embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
    self.pos_Embedding = nn.Embedding(self.maxLen, self.embedding_dim)

    self.norm = nn.LayerNorm(self.embedding_dim)

  def forward(self, inp):

    pos = torch.arange(self.maxLen, dtype=torch.long).to(device)
    pos = pos.unsqueeze(0).expand_as(inp)

    embedding = self.inp_Embedding(inp) + self.pos_Embedding(pos)
    return self.norm(embedding)


class Encoder(nn.Module):
  def __init__(self, embedding_dim, dropout_size, number_heads):
    super(Encoder, self).__init__()

    #variable initialization
    self.number_heads = number_heads
    self.embedding_dim = embedding_dim
    self.dropout_size = dropout_size

    self.multi_head_attn = MultiHeadAttention(self.number_heads, self.embedding_dim)

    self.feed_forward = nn.Sequential(
        nn.Linear(self.embedding_dim, self.embedding_dim),
        nn.Dropout(self.dropout_size),
        nn.GELU(),
        nn.Linear(self.embedding_dim, self.embedding_dim),
        nn.Dropout(self.dropout_size)
    )

    self.norm = nn.LayerNorm(self.embedding_dim)  

  def forward(self, inps, attn_mask):
    
    # print("Encoder")
    
    inps = inps + self.multi_head_attn(inps, attn_mask)
    
    # print("inps.shape after multi_head: ", inps.shape)
    
    inps = inps + self.feed_forward(inps)
    
    # print("inps.shape after feedforward: ", inps.shape)

    return self.norm(inps)


class MultiHeadAttention(nn.Module):
  def __init__(self, number_heads, hidden_dim):
    super(MultiHeadAttention, self).__init__()

    self.number_heads = number_heads
    self.hidden_dim = hidden_dim

    self.attention_heads = nn.ModuleList([
      AttentionHead(self.hidden_dim, self.hidden_dim)
      for _ in range(self.number_heads)
      ])

    self.ln = nn.Linear(self.number_heads * self.hidden_dim, self.hidden_dim)
    self.norm = nn.LayerNorm(self.hidden_dim)  

  def forward(self, inp, attn_mask):
    
    # print("MultiHeadAttention")
    
    scores = [head(inp, attn_mask) for head in self.attention_heads]
    
    # print("scores.shape after heads: ", len(scores), len(scores[0]))
    
    
    scores = torch.cat(scores, dim=-1)
    
    # print("scores.shape after concat: ", scores.shape)
    
    
    scores = self.norm(self.ln(scores)+inp)
    
    # print("scores.shape after norm(len): ", scores.shape)
    
    
    return scores


class AttentionHead(nn.Module):
  def __init__(self, hidden_dim, number_heads):
    super(AttentionHead, self).__init__()
    self.hidden_dim = hidden_dim
    self.number_heads = number_heads

    self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.value = nn.Linear(self.hidden_dim, self.hidden_dim)


  def forward(self, inp, attn_mask):
    
    # print("AttentionHead")
    
    # print("inp shape: ", inp.shape)
    # print("self.hidden_dim: ", self.hidden_dim)
    
    k = self.key(inp)
    q = self.query(inp)
    v = self.value(inp)
    
    # print("k shape: ", k.shape)
    # print("q shape: ", q.shape)
    # print("v shape: ", v.shape)

    scores = torch.bmm(q, k.transpose(1, 2))/math.sqrt(q.size(1))
    scores = scores.masked_fill_(attn_mask, float("-inf"))
    
    # print("scores shape in head after bmm, masked_fill: ", scores.shape)

    attn = nn.Softmax(dim=-1)(scores)
    
    # print("attn shape: ", attn.shape)
    
    ctx = torch.bmm(attn, v)
    
    # print("ctx.shape:", ctx.shape)
       
    return ctx


class Generator(nn.Module):
  def __init__(self, vocab_size, maxLen, embedding_dim, dropout, nbr_layers, nbr_heads):
    super(Generator, self).__init__()
    
    self.vocab_size = vocab_size
    self.maxLen = maxLen
    self.embedding_dim = embedding_dim
    self.dropout = dropout
    self.nbr_heads = nbr_heads
    self.nbr_layers = nbr_layers

    self.embedding = CustomEmbedding(self.vocab_size, self.embedding_dim, self.maxLen)

    self.enc_layers = nn.ModuleList([
      Encoder(self.embedding_dim, self.dropout, self.nbr_heads)
      for _ in range(self.nbr_layers)
      ])
    
    self.fc_word = nn.Linear(self.embedding_dim, self.vocab_size)

    self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
    self.activ = nn.ReLU()
    self.norm = nn.LayerNorm(self.embedding_dim)
    self.embed_weight = self.embedding.inp_Embedding.weight
    self.n_vocab, self.n_dim = self.embed_weight.size()
    self.decoder = nn.Linear(self.n_dim, self.n_vocab, bias=False)
    self.decoder.weight = self.embed_weight
    self.decoder_bias = nn.Parameter(torch.zeros(self.n_vocab))


  def forward(self, inp, masked_inp, attn_mask):


    #embedding(inp).shape = (batch_size, maxLen, embedding_dim)
    # print('inp.shape: ', inp.shape)
    # print('attn_mask.shape: ', attn_mask.shape)
    
    inp = self.embedding(inp)
    
    # print('inp after embedding shape: ', inp.shape)

    for enc_layer in self.enc_layers:
      inp = enc_layer(inp, attn_mask)
    
    

    # masked_inp = masked_inp[:, :, None].expand(-1, -1, inp.size(-1))
    
    # masked_inp = masked_inp.type(torch.int64) 
    # masked = torch.gather(inp, 1, masked_inp)
    # print("masked: ", masked.argmax(-1))
    masked = self.norm(self.activ(self.linear(inp)))
    inp =  self.decoder(masked) + self.decoder_bias
    #predcition of word indices
    return inp#self.fc_word(inp)
