

import os
from datasets import load_dataset
import numpy as np


class TinyStoriesTokenizer:
    def __init__(self):
        self.vocab_size = 256
    # 标准 ASCII 只有 128 个字符，id 范围是 0–127。
    # 因此 TinyStoriesTokenizer 中 vocab_size = 256 没用全

    def encode(self, txt):
        # 判断输入是否为单个字符串。
        if isinstance(txt, str):
            # 每个 字符 变成 其对应的ascii数值
            tmp = txt.encode('ascii', errors="ignore")
            # 把这串数字原样装进一个 NumPy 数组里,每个数字成为数组的一个元素,类型是 np.uint8。 其结果是一个np的list
            x = np.frombuffer(tmp, dtype=np.uint8)

        # 是否为列表 (列表中每个元素都是 string)
        elif isinstance(txt, list):
            # 每个元素内容变成 np的list，字符也变成ascii值。
            x = [np.frombuffer(t.encode('ascii', errors="ignore"), dtype=np.uint8) for t in txt]
        return x

    def decode(self, x):
        assert isinstance(x, np.ndarray)
        x = x.astype(np.uint8)
        if x.ndim == 1:
            txt = x.tobytes().decode('ascii', errors="ignore")
        elif x.ndim == 2:
            txt = [t.tobytes().decode('ascii', errors="ignore") for t in x]
        return txt

def prepare_dataset(dataset_path):
    # 将所有文本头尾相连成一个文本。文本内容每个字符串 变成 其对应的ascii值，将这段文本放入np的array中，保存到本地 .npy 文件中


    # 1. 加载/下载 数据集
    ds_train = load_dataset("roneneldan/TinyStories", split="train", streaming=False)
    ds_test = load_dataset("roneneldan/TinyStories", split="validation", streaming=False)
    # ds_train 本身是一个 Dataset 对象，结构上类似一张表，每一行是一条样本
    # ┌────────────────────────────┐
    # │ row 0: {"text": "story1..."} │
    # │ row 1: {"text": "story2..."} │
    # │ row 2: {"text": "story3..."} │

    # 2. 取出文本列，转成 list
    # ds_train['text']  →  ['story1文本', 'story2文本', ..., 'storyN文本']
    ds_train_list = list(ds_train['text'])
    ds_test_list = list(ds_test['text'])


    tokenizer = TinyStoriesTokenizer()

    # 3. 逐条编码（tokenizer.encode 传入 list 分支）
    x_train = tokenizer.encode(ds_train_list)
    x_test = tokenizer.encode(ds_test_list)

    # x_train = [
    #     array([72, 105, 33, ...], dtype=uint8),   # story1 的每个字符→ASCII码
    #     array([84, 104, 101, ...], dtype=uint8),  # story2 的每个字符→ASCII码
    #     ...
    # ]

    # 4. 拼接成一个长文本
    # np.concatenate 的第一个参数是一个序列(list、tuple 都行),序列里的每个元素是数组或能转成数组的对象（list 里放普通 Python list 也行,NumPy 会先把它们转成数组）
    x_train = np.concatenate(x_train)
    x_test = np.concatenate(x_test)
    # 拼接前：
    # x_train = [
    #     array([72, 105, 33], dtype=uint8),        # story1, 长度3
    #     array([84, 104, 101, 32, ...], dtype=uint8), # story2, 长度可能是500
    #     array([79, 110, 99, 101, ...], dtype=uint8), # story3, 长度可能是320
    #     ...
    # ]

    # 拼接后：
    # x_train = np.concatenate(x_train)
    # # → array([72, 105, 33, 84, 104, 101, 32, ..., 79, 110, 99, 101, ...], dtype=uint8)

    # 所有文本头尾相连成一个文本。
    # 核心逻辑是：如果 block_size（训练窗口，比如256或1024字节）远小于平均故事长度，那么绝大多数采样出来的窗口，其内容还是落在同一个故事内部，真正跨越两个故事边界的窗口只占少数。
    # 像 Andrej Karpathic 风格的 TinyStories 复现（这份代码明显是仿照这类实现的）大多确实是直接拼接、不加分隔符，因为字节级/字符级模型对这种噪声容忍度较高，而且这类小模型实验更关心"语法连贯性"这种粗粒度指标，不太在意这种边界瑕疵。
    # 像 GPT 系列实际预训练用的做法，通常会在文档之间插入一个特殊 token（比如 <|endoftext|>）

    # 如果你在意这个问题，可以怎么改
    # 在每条故事之间插入一个分隔符，比如用一个不会在正常文本中出现的字节
    # 例如用 0（NUL）或者专门在 encode 里保留一个 id 给 <eos>
    # x = [np.frombuffer(t.encode('ascii', errors="ignore") + b'\x00', dtype=np.uint8) for t in txt]

    # x_train.shape == (1,)
    # np.concatenate 结果是一 numpy 数组（ndarray）(类似list)，
    # 原本 list 里有210万个"子数组"，拼接后只剩一个数组，包含所有文字

    # .npy 是 NumPy 专门为 ndarray 设计的二进制序列化格式
    os.makedirs(dataset_path, exist_ok=True)
    np.save(f"{dataset_path}/train.npy", x_train)
    np.save(f"{dataset_path}/test.npy", x_test)


if __name__ == "__main__":
    # login(token=os.environ['HF_TOKEN'])
    prepare_dataset(dataset_path="./tinystories/")

