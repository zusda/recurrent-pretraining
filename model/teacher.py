
import torch.nn as nn
from dataclasses import dataclass, field
from x_transformers import Encoder, Decoder
import torch.nn.functional as F
import torch
from einops import rearrange, repeat

from .tools import *


@dataclass
class TeacherConfig:
    vocab_size: int = 256 # vocab_size：词表大小，即 token id 的取值范围是 0 ~ vocab_size-1
    d_model: int = 256 # 模型向量维度


    mem_vecs: int = 16 # mem 的向量数量
    mem_total_size: int = mem_vecs * d_model # the total memory state size, Z

    encoder_depth: int = 8
    decoder_depth: int = 4

    n_heads: int = 8 # the number of attention heads



class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = TeacherConfig()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.encoder = Encoder(
            dim=cfg.d_model,            # 向量维度
            depth=cfg.encoder_depth,    # 层数
            heads=cfg.n_heads,          # 头数
            rotary_pos_emb=True,   # 旋转位置编码（RoPE）
            attn_flash=True,
            use_rmsnorm=True,       # 用 RMSNorm 代替 LayerNorm （Encoder 内部所有归一化层都换成 RMSNorm）

            # LayerNorm：减均值、除标准差，再做缩放和平移：
            # RMSNorm：只除以均方根，不减均值，也没有偏置 β：
            # 为什么用它：
            #   计算更简单、更快（省掉均值计算和偏置）
            #   实验表明效果与 LayerNorm 基本持平，训练同样稳定

        )

        self.decoder = Decoder(
            dim=cfg.d_model,
            depth=cfg.decoder_depth,
            heads=cfg.n_heads,
            rotary_pos_emb=True,
            use_rmsnorm=True,
            attn_flash=True,)

        # 线性映射翻译层  vec -> token_logit。 经过softmax等方法采样后才成为token，但训练时不用解码策略。
        self.to_logits = nn.Linear(cfg.d_model, cfg.vocab_size)

        self.encoder_mem_register = nn.Embedding(cfg.mem_vecs, cfg.d_model) # the input register tokens for memory tokens for the encoder

    def encode(self, x_ctx):
        B, seq = x_ctx.shape

        # 获取mask
        mask = self.get_mask(x_ctx)

        # 转成向量
        x_emb = self.embed(x_ctx)


        mem_emb = self.encoder_mem_register(
            torch.arange(
                self.cfg.mem_vecs,
                device=self.encoder_mem_register.weight.device
                # torch.arange 默认在 CPU 上创建 索引 ；
                # nn.Embedding 查表要求 索引 与 向量 在同一设备，否则会报设备不一致的错误。
                # 因此这里让索引直接创建在 encoder_mem_register 内部向量所在的设备上。
            )
        )


        # 把 (K, D) 沿着 batch 维广播复制成 (B, K, D) —— 每个样本用同一组 memory token 初始值
        mem_emb = repeat(mem_emb, "K D -> B K D", B=B)


        # 将两份做好的向量拼接起来（mem_emb 放最后面）
        encoder_input = torch.cat([x_emb, mem_emb], dim=1)

        # 将输入送入encoder网络中，获得输出z (B, seq+K, D)
        # z = self.encoder(encoder_input, mask=mask)
        # mask是为了attention 有效 token 不与padding等位置的token进行联系，而我们输入的数据是不存在的。
        z = self.encoder(encoder_input)

        # 仅提取mem部分token计算结果
        z = z[:, -self.cfg.mem_vecs:, :] # (B, K, D)


        # 将每个mem的数值归一化，减小不同mem的数量级差距
        z = normalize_mem(z, rms_norm=True)



        # 把 (B, K, D) 展平成 (B, K*D)，即把 K 个记忆槽位的向量首尾拼接成一条长向量
        z = rearrange(z, "B K d -> B (K d)") # (B Z)

        # encoder 将 context作为输入， 将压缩后的 mem vector 结果作为输出
        return z



    def get_mask(self,x_ctx):
        # Transformer 所需的 mask 是 torch.Tensor，dtype 为 torch.bool，形状 (B, N)
        # 即每个位置为bool值的torch.Tensor

        # x_ctx (B,seq) 每个位置token值
        B, seq = x_ctx.shape

        mask_x = (x_ctx != -1)  # -1 是 padding 位置上的值

        x_ctx = x_ctx.clamp(min=0) # -1 是无效的 embedding 索引（会越界报错），先把所有负数（也就是padding位）强制截断成0

        # mem 的 mask。 全部标记为 True
        mask_mem = torch.ones(
            (B, self.cfg.mem_vecs),
            dtype=torch.bool,           # dtype=torch.bool，torch.ones 生成的实际值是 True，不是整数 1
            device=mask_x.device
        )

        # 我们固定mem位置放在最后面
        mask = torch.cat([mask_x, mask_mem], dim=1)

        return mask


    def decode(self, z, x_fut):
        # encode() 最后把 (B, K, D) 展平成了 (B, K*D)，这里做逆操作，还原回 (B, K, D)
        z = rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_vecs)

        # mem token拼接在前面，当作输入
        decoder_input = torch.cat([z, self.embed(x_fut)], dim=1) # (B,K+seq,D)

        # decoder 默认使用 causal mask（因果掩码），那么每个位置只能关注自己和前面的 token，不能关注后面的 token。不用自己加mask。
        output = self.decoder(decoder_input) # (B,K+seq,D)

        # 目标预测位置是 [ self.cfg.mem_vecs : ]
        # 对目标位置的对应输出位置，是目标位置整体前移一格位置的输出：
        # [self.cfg.mem_vecs-1:-1] ，最后一个mem的输出，以及排除最后一个输入位置的输出
        logits = self.to_logits(output)[:, self.cfg.mem_vecs-1:-1, :]  # # (B,seq,vocab_size)

        return logits




























