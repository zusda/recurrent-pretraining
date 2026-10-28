
import torch
from torch.utils.data import Dataset
import numpy as np



class MyDataset():
    def __init__(self, dataset_path, tokenizer):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer

        self.data = np.load(f"{self.dataset_path}") # 一条长的ascii值np array，仅有1维，长度为n


    def sample_batch(self, batch_size: int, sample_len: int):

        """
        Sample a batch of data from the dataset.
        Returns a dictionary with keys 'x' and 'y'.
        x: (batch_size, ctx_len) of input tokens
        y: (batch_size, ctx_len) of output target tokens.
        In this case, y[t] = x[t+1]
        """
        len_ = len(self.data) - sample_len - 1

        # 随机生成batch_size个起始下标
        idxs = np.random.randint(0, len_, batch_size)

        # 对每个随机起点i，切出长度为ctx_len+1的一段。（比正常序列长度多1个token，因为要同时构造出 x 和 y，而 y 是 x 整体右移一位得到的)
        # 再把batch_size段堆叠成一个二维数组，形状 (batch_size, ctx_len+1)。
        batch = np.stack([self.data[i:i+sample_len+1] for i in idxs])


        # 转成PyTorch张量
        # batch = torch.from_numpy(batch).long()
        # 保持原 dtype，不调用 .long()
        batch = torch.from_numpy(batch)
        # 因为 .long() 会把数据放大 8 倍（uint8 → int64），而这个放大完全可以推迟到 GPU 上做，那里几乎不花时间。


        # batch[:, :-1]：去掉最后一个token，形状 (batch_size, ctx_len)，作为输入x
        # batch[:, 1:]：去掉第一个token，形状 (batch_size, ctx_len)，作为目标y
        return dict(x=batch[:, :-1], y=batch[:, 1:])



from .tinystories_prepare import TinyStoriesTokenizer

import os
def get_tinystories_dataset():
    tokenizer = TinyStoriesTokenizer()
    # dataset_path = "./tinystories/"
    # print(os.getcwd())

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "tinystories")
    # print(dataset_path)

    train_ds = MyDataset(os.path.join(dataset_path, "train.npy"), tokenizer)
    test_ds = MyDataset(os.path.join(dataset_path, "test.npy"), tokenizer)

    return train_ds,test_ds,tokenizer




