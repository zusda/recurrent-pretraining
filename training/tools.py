



import pickle
def load_pkl_path(load_path):
    if load_path is not None:
        with open(load_path, "rb") as f:
            return pickle.load(f)
    else:
        return None


def save_pkl_path(save_path, item):
    if save_path is not None:
        # os.makedirs(save_path, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(item, f)




import queue
import threading

class Prefetcher:
    # 正常训练中，先读取batch数据，然后再用该数据进行训练。 但是读取batch本身可能也非常花时间

    # 该类使用多线程预先读取 batch 数据，存放到queue中。当训练需要输入数据的时候，使用get()获取queue中的数据。
    # 如果queue没数据，q.get() 会自动阻塞，直到有数据为止。

    def __init__(self, ds, batch_size, ctx_len, depth=4):
        self.q = queue.Queue(maxsize=depth)   # 最多缓存 depth 个 batch，满了 put 会自动阻塞
        def work():
            while True:                        # 无限循环：采样 → 放入队列
                self.q.put(ds.sample_batch(batch_size, ctx_len))
        threading.Thread(target=work, daemon=True).start()  # daemon：主程序退出时线程自动结束

    def get(self):
        batch = self.q.get()                   # 队列空了会等；有就直接拿
        return {k: v.cuda(non_blocking=True).long() for k, v in batch.items()}


import math
def get_lr(itr, n_iters, lr, lr_schedule):

    # 不管训练到第几步，系数恒为1，学习率全程保持lr不变。
    if lr_schedule == "constant":
        coeff = 1.


    elif lr_schedule == "warmup_constant":
        warmup_iters = n_iters // 100
        # coeff从0线性增长到1
        if itr < warmup_iters:
            coeff = itr / warmup_iters

        # warmup结束后：coeff恒为1，学习率保持lr不变，直到训练结束
        else:
            coeff = 1.



    elif lr_schedule == "warmup_cosine": # Linear warmup followed by cosine decay
        warmup_iters = n_iters // 100

        # warmup阶段：和上面一样，线性从0升到1
        if itr < warmup_iters:
            coeff = itr / warmup_iters


        else:
            # 把" warmup结束后的 剩余训练轮数 " 归一化到[0,1]区间
            decay_ratio = (itr - warmup_iters) / (n_iters - warmup_iters)
            assert 0 <= decay_ratio <= 1
            # 中间过程 按余弦曲线 平滑下降
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))


    return coeff * lr



def get_smt_batch(batch, Tc, Tf):
    # batch数据中，切出需要使用的 x_ctx, x_fut, y_fut, x_ctx_next


    # 取出完整序列
    x, y = batch['x'], batch['y']


    # 取序列最开头的Tc个token作为context (从0开始算，所以ctx实际上第Tc-1才是最后一个token)
    x_ctx = x[:, :Tc]

    # 紧接着context后面的Tf个token作为future
    x_fut = x[:, Tc:Tc+Tf]

    # y_fut = y[:, Tc-1:Tc+Tf-1]（[y_3, y_4, y_5, y_6]）
    # 在数值上等价于
    # x_fut = x[:, Tc:Tc+Tf]    （[x_4, x_5, x_6, x_7]）
    # ——本质上是同一组token
    # 只是通过 y 这个已经错位好的变量取出来，方便直接拿去和 decode 输出的logits做交叉熵loss。
    y_fut = y[:, Tc-1:Tc+Tf-1]

    x_ctx_next = x[:, 1:Tc+1]


    # 举例 Tc=4时
    # x_ctx = x[:, :Tc]           # Tc=4时 →      [x_0, x_1, x_2, x_3]
    # x_fut = x[:, Tc:Tc+Tf]        # Tc=4时 →    [x_4, x_5, x_6, x_7]
    # y_fut = y[:, Tc - 1:Tc + Tf - 1]            [y_3, y_4, y_5, y_6]
    # [x_0, x_1, x_2, x_3] 被算进 x_ctx 里去生成    m_3
    # x_ctx_next = x[:, 1:Tc+1]                   [x_1, x_2, x_3, x_4]

    return x_ctx, x_fut, y_fut, x_ctx_next


import torch.nn.functional as F
from einops import rearrange
def calc_cross_entropy_loss(logits, y):
    # logits 每个位置的logit输出，形状为 (B, T, V)，  T：序列长度  V：词表大小
    # y 每个位置对应的真实目标token id，形状 (B, T)，每个位置是一个整数


    # 交叉熵loss的优化方向：让 logits 这个长度V的向量里，第y个位置对应的值相对更高


    # F.cross_entropy 原生的输入要求
    # logits: (B, category)  —— 必须是二维
    # y:      (B,)     —— 必须是一维
    # 其输出结果也是   (B,)


    # 这份代码里，logits 常常是 (B, T, V)（三维），或者其他场合可能是别的维度组合，直接传进去会报错或维度不匹配。
    # 这个函数就是负责"展平成PyTorch要求的形状 → 算loss → 再还原回原来的形状"。

    # 先记录 y 原本的形状（比如 (B, T)）
    shape = y.shape

    # 除了最后一维V之外的所有维度全部压扁合并成一个维度：
    # (B, T, V) → (B*T, V)
    # (B, T) → (B*T,)
    logits, y = rearrange(logits, '... V -> (...) V'), rearrange(y, '... -> (...)')

    # 因为用了 reduction='none' 并且最后做了 .reshape(shape)，所以最终返回的loss和 y 的维度完全一致(B, T)
    return F.cross_entropy(logits, y, reduction='none').reshape(shape)


import torch
def calc_uniformity_loss(z, t=2, sample=512):
    # uniformity loss 衡量一批向量在单位超球面上分布得是否均匀（越均匀说明表示越不容易塌缩成几个相似的点）
    # 每个向量是一个球面上的点
    # uniformity loss值小（更负） → 点在球面上分布得越均匀分散，彼此越不趋同，区分度越高
    # uniformity loss值大（接近0） → 点越聚在一起/趋同（同一化程度高），区分度越低

    # sample=512：每次只随机抽取最多512个向量
    # t=2：温度系数（temperature）控制"距离"对最终loss的敏感程度/放大倍数

    # 输入 z：(..., D)，任意前置维度 + 最后一维是特征维D
    # 输出 一个标量（0维张量，一个数值）
    """
    Calculate the uniformity loss between the memory tokens.
    z: (..., D)
    t: float
    sample: int (number of samples to draw for computing the uniformity loss)
    Returns: float
    """


    z = z / (z.norm(dim=-1, keepdim=True) + torch.finfo(z.dtype).eps)
    z = rearrange(z, "... D -> (...) D")
    idx = torch.randperm(len(z))[:sample]
    z = z[idx]
    return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().log()

# import os
#
# def save_pkl(save_dir, name, item):
#     # 通用的"把任意Python对象序列化保存成.pkl文件"的工具函数
#     if save_dir is not None:
#         os.makedirs(f"{save_dir}/", exist_ok=True)
#         with open(f"{save_dir}/{name}.pkl", "wb") as f:
#             pickle.dump(item, f)
#
# def load_pkl(load_dir, name):
#     if load_dir is not None:
#         with open(f"{load_dir}/{name}.pkl", "rb") as f:
#             return pickle.load(f)
#     else:
#         return None
#



def get_dmt_batch(batch, t, cfg):
    x, y = batch['x'], batch['y']

    x_ctx_start = t
    x_ctx_end = t + cfg.T_ctx

    x_fut_start = t + cfg.T_ctx
    x_fut_end = t + cfg.T_ctx + cfg.T_fut

    y_fut_start = t + cfg.T_ctx - 1
    y_fut_end = t + cfg.T_ctx + cfg.T_fut - 1

    x_ctx = x[:, x_ctx_start:x_ctx_end]
    x_fut = x[:, x_fut_start:x_fut_end]
    y_fut = y[:, y_fut_start:y_fut_end]

    return x_ctx, x_fut, y_fut









