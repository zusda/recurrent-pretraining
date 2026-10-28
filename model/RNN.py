


import torch.nn as nn
from dataclasses import dataclass, field
from x_transformers import Encoder, Decoder
import torch.nn.functional as F
import torch
from einops import rearrange, repeat

from .tools import *





@dataclass
class RNNConfig:

    vocab_size: int = 256 # vocab_size：词表大小，即 token id 的取值范围是 0 ~ vocab_size-1
    d_model: int = 256 # 模型向量维度

    mem_vecs: int = 16 # mem 的向量数量
    mem_total_size: int = mem_vecs * d_model # the total memory state size, Z

    encoder_depth: int = 8
    decoder_depth: int = 4

    n_heads: int = 8 # the number of attention heads


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = RNNConfig()
        self.cfg = cfg


        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rnn = Encoder(
            dim=cfg.d_model,
            depth=cfg.encoder_depth,
            heads=cfg.n_heads,
            rotary_pos_emb=True,
            use_rmsnorm=True,
            attn_flash=True,
        ) # RNN's Transformer blocks

        self.decoder = Decoder(
            dim=cfg.d_model,
            depth=cfg.decoder_depth,
            heads=cfg.n_heads,
            rotary_pos_emb=True,
            use_rmsnorm=True,
            attn_flash=True,
        )  # decoder

        self.to_logits = nn.Linear(cfg.d_model, cfg.vocab_size)


    #  rnn_update(mem,x) -> mem_next
    def rnn_update(self, z, x):
        # 把展平的记忆状态还原成 (B, K, D)
        z = rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_vecs)

        # 把 mems 和 新的输入x 拼接起来, mems 放前面
        inputs = torch.cat([z, self.embed(x)], dim=1) # (B, K+T, D)

        output = self.rnn(inputs) # (B, K+T, D)

        # 只取前K个位置（对应原本 mems 的位置）的输出
        z_next = output[:, :self.cfg.mem_vecs, :] # (B, K, D)

        # 每个mem 进行归一化
        z_next = normalize_mem(z_next) # (B Z)
        z_next = rearrange(z_next, "B K d -> B (K d)") # (B Z)

        return z_next



    def rnn_readout(self, z): # (B, Z)

        # 记忆还原成 (B, K, D)。
        z = rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_vecs)  # (B,K,D)

        # 取最后一个位置的输出，该位置预测下一token
        o = self.decoder(z)[:, -1, :]  # (B,D)

        logits = self.to_logits(o) # (B, vocab_size)

        return logits # (B, V)










