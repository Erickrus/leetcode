# Pytorch transformers from scratch (Aladdin Persson)
# https://www.youtube.com/watch?v=U0s0f995w14&t=2735s


# questions, uncertain stuffs
# 1. How Nx is used for encoder and decoder stack
# 2. Why transformer block takes the same x as k q v


# Some Basics
# torch.nn building blocks
# https://pytorch.org/docs/stable/nn.html
#  Containers: Module, ModuleList, Sequential
#  Convolution Layers: Conv2d
#  Pooling Layers: MaxPool2d
#  Activations: ReLU
#  Normalization Layers: LayerNorm
#  Linear Layer: Linear
#  Dropout Layers: Dropout
#  Sparse Layers: Embedding

# https://pytorch.org/docs/stable/torch.html
#  Tensors:
#    numel()
#  Creation Ops:
#    tensor(), arange(), randn(), ones(), zeros(), from_numpy()
#  Indexing, Slicing, Joining, Mutating Ops:
#    cat(), squeeze(), unsqueeze(), transpose(), stack()
#  BLAS and LAPACK Operations:
#    mm(), bmm(), matmul()
#  Other Operations:
#    tril(), triu(), einsum()
# torch.sparse.softmax()


#torch.Tensor
#  descriptors
#    shape, numel(), size(), type(), dim()
#  shaping
#    reshape(), squeeze(), unsqueeze(), transpose(), select(), expand(), permute()
#  conversion
#    to(), numpy()
#  stats
#    sum()
#  misc
#    masked_fill()

#pil conversions
#  torchvision.transforms.functional
#    to_pil_image(), to_tensor()


# Class dependencies
#
#               Transformer
#                 @Encoder, @Decoder
#                  /            \
# Encoder         /          Decoder
#   @TransformerBlock x N      @DecoderBlock x N
#   Embedding, Dropout         Embedding, Linear, Dropout
#         |                      |
#         |                  DecoderBlock
#         |                 /  @SelfAttention, @TransformerBlock
#         |         /------/   LayerNorm, dropout
#         |        /                    |
# TransformerBlock                      |
#   @SelfAttention                      |
#   LayerNorm, Linear, Dropout, ReLU   /
#                      \              /
#                       \            /
#                       SelfAttention
#                         Linear

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
  def __init__(self, embed_size, heads):
    super(SelfAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dim = embed_size // heads

    assert (self.head_dim * heads == embed_size), "Embed size need to be div by heads"

    # torch.nn.Linear Documentation
    # bias – If set to False, the layer will not learn an additive bias
    # y = x * A.transpose
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

    self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

  def forward(self, values, keys, queries, mask):
    N = queries.shape[0] # How many example to send in the same time
    value_len, key_len, query_len = values.shape[1],  keys.shape[1], queries.shape[1]

    # Split embedding into self.heads pieces
    values = values.reshape(N, value_len, self.heads, self.head_dim)
    keys = keys.reshape(N, key_len, self.heads, self.head_dim)
    queries = queries.reshape(N, query_len, self.heads, self.head_dim)

    # Einsum is all you need (Aladdin Persson)
    # https://www.youtube.com/watch?v=pkVwUVEHmfI
    # Exact explanations of einsum
    
    # Flatten, Reshape, and Squeeze Explained (deeplizard)
    # https://www.youtube.com/watch?v=fCVuiW9AFzY&t=84s
    # Pytorch permute函数 (胡孟)
    # https://zhuanlan.zhihu.com/p/76583143


    values = self.values(values)
    keys = self.keys(keys)
    queries = self.queries(queries)
    

    # 此处energy就是公式里的 $QK^T$
    # energy calculation
    # queries shape: (N, query_len, heads, heads_dim)
    # keys shape: (N, query_len, heads, heads_dim)
    # energy shape: (N, heads, query_len, key_len)
    energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

    # 等价于 torch.matmul(queries, keys.transpose(-2, -1))

    # How to code The Transformer in Pytorch 中的 score有多种意思, 这个变量被反复重定义了
    # 在最外层的scores说的是attention计算得到的值
    # https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

    if mask is not None:
      # Here, mask is a tensor with the same shape as energy
      # masked_fill method, explains how it works
      # https://programmersought.com/article/68273730436/
      # replace mask==0, where element is True, to -1e20
      energy = energy.masked_fill(mask==0, float("-1e20"))

    # $Attention(Q, K, V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$
    # normalizes values along axis 3 (head_dim), within each head
    attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

    out = torch.einsum("nhql,nlhd->nqhd",[attention, values]).reshape(
      N, query_len, self.heads*self.head_dim
    )
    # attention shape: (N, heads, query_len, key_len)
    # values shape: (N, value_len, heads, heads_dim)
    # (N, query_len, heads, head_dim)
    # after einsum, then flattern last 2 dim

    out = self.fc_out(out)
    return out

class TransformerBlock(nn.Module):
  def __init__(self, embed_size, heads, dropout, forward_expansion):
    super(TransformerBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    # feed_forward is a simple magnify and shrink process
    # with a relu activation in the middle
    # 本例中forward_expansion 4倍
    self.feed_forward = nn.Sequential(
      nn.Linear(embed_size, forward_expansion*embed_size),
      nn.ReLU(),
      nn.Linear(forward_expansion*embed_size, embed_size)
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, value, key, query, mask):
    attention = self.attention(value, key, query, mask)

    x = self.dropout(self.norm1(attention + query))
    forward = self.feed_forward(x)
    out = self.dropout(self.norm2(forward+x))
    return out

class Encoder(nn.Module):
  def __init__(
    self,
    src_vocab_size,
    embed_size,
    num_layers,
    heads,
    device,
    forward_expansion,
    dropout,
    max_length
  ):
    super(Encoder, self).__init__()
    self.embed_size = embed_size
    self.device = device
    self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
    self.position_embedding = nn.Embedding(max_length, embed_size)

    self.layers = nn.ModuleList(
      [
       TransformerBlock(
         embed_size,
         heads,
         dropout=dropout,
         forward_expansion=forward_expansion
       ) for _ in range(num_layers)
      ]
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    N, seq_length = x.shape
    positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

    out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
    for layer in self.layers:
      out = layer(out, out, out, mask)

    return out

class DecoderBlock(nn.Module):
  def __init__(
    self,
    embed_size,
    heads,
    forward_expansion,
    dropout,
    device
  ):
    super(DecoderBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm = nn.LayerNorm(embed_size)
    self.transformer_block = TransformerBlock(
      embed_size, heads, dropout, forward_expansion
    )
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x, value ,key, src_mask, tgt_mask):
    attention = self.attention(x, x, x, tgt_mask)
    query = self.dropout(self.norm(attention +x))
    out = self.transformer_block(value, key, query, src_mask)
    return out

class Decoder(nn.Module):
  def __init__(self,
    tgt_vocab_size,
    embed_size,
    num_layers,
    heads,
    forward_expansion,
    dropout,
    device,
    max_length):
    super(Decoder, self).__init__()
    self.device = device
    self.word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
    self.position_embedding = nn.Embedding(max_length, embed_size)

    self.layers = nn.ModuleList(
      [
       DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
       for _ in range(num_layers)
      ]
    )

    self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x, enc_out,src_mask, tgt_mask):
    N, seq_length = x.shape
    positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
    x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

    for layer in self.layers:
      x = layer(x, enc_out, enc_out, src_mask, tgt_mask)
    
    out = self.fc_out(x)

    return out

class Transformer(nn.Module):
  def __init__(self,
    src_vocab_size, 
    tgt_vocab_size, 
    src_pad_idx, 
    tgt_pad_idx, 
    embed_size=256,
    num_layers = 6, 
    forward_expansion=4, 
    heads=8,
    dropout=0, 
    device="cpu",
    max_length=100
  ):
    super(Transformer, self).__init__()
    self.encoder = Encoder(
      src_vocab_size,
      embed_size,
      num_layers,
      heads,
      device,
      forward_expansion,
      dropout,
      max_length 
    )
    self.decoder = Decoder(
      tgt_vocab_size,
      embed_size,
      num_layers,
      heads,
      forward_expansion,
      dropout,
      device,
      max_length 
    )

    self.src_pad_idx = src_pad_idx
    self.tgt_pad_idx = tgt_pad_idx
    self.device = device

  def make_src_mask(self, src):
    src_mask = (src!= self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    # (N, 1, 1, src_len)
    return src_mask.to(self.device)

  def make_tgt_mask(self, tgt):
    N, tgt_len = tgt.shape
    tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(
        N, 1, tgt_len, tgt_len
    )
    # (N, 1, tgt_len, tgt_len)
    return tgt_mask.to(self.device)

  def forward(self, src, tgt):
    src_mask = self.make_src_mask(src)
    tgt_mask = self.make_tgt_mask(tgt)

    enc_src = self.encoder(src, src_mask)
    out = self.decoder(tgt, enc_src, src_mask, tgt_mask)

    return out


if __name__ == "__main__":
  device = "cpu"
  x = torch.tensor([
    [1,5,6,4,3,9,5,2,0],
    [1,8,7,3,4,5,6,7,2],
    ]).to(device)

  tgt = torch.tensor([
    [1,7,4,3,5,9,2,0],
    [1,5,6,2,4,7,6,2],
    ]).to(device)

  src_pad_idx =0
  tgt_pad_idx =0
  src_vocab_size=10
  tgt_vocab_size=10

  model = Transformer(src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx).to(device)
  out = model(x, tgt[:,:-1])
  print(out.shape)

