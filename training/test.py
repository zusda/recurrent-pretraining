import os

import torch
import numpy as np

from model.RNN import RNN
from model.teacher import Teacher
from data.data_wrapper import get_tinystories_dataset

# ---------------- 配置 ----------------
save_dir = r"F:\code\python\RNN_with_smt_dmt\runs\dmt_tinystories_0"  # 新训练的 DMT 目录（rnn.pt、teacher.pt 所在处）
device = "cuda"                          # 模型和数据放的设备
num_samples = 5                          # 取几条测试数据
prompt_len = 256                         # 提示长度，必须等于训练用的 T_ctx=256，因为初始记忆来自 teacher 对满 256 字节的编码
gen_len = 200                            # 续写多少个 token
temperature = 0.8                        # 0 为贪心；0.7~1.0 为随机采样（贪心容易重复，先用 0.8）
token_key = "x"                          # batch 里存输入 token id 的键
prompt_show_chars = 100                  # 提示太长，只打印结尾这么多字符


def decode_text(tokenizer, ids):
    array = ids.cpu().numpy().astype(np.uint8)  # GPU 张量 -> CPU -> NumPy 数组，每个值转成 0~255 的字节
    return tokenizer.decode(array)


def load_models():
    rnn = RNN()
    rnn.load_state_dict(torch.load(os.path.join(save_dir, "rnn.pt"), map_location="cpu", weights_only=False))
    rnn.to(device)
    rnn.eval()

    teacher = Teacher()
    teacher.load_state_dict(torch.load(os.path.join(save_dir, "teacher.pt"), map_location="cpu", weights_only=False))
    teacher.to(device)
    teacher.eval()
    return rnn, teacher


def get_test_ids(test_ds, count, total_len):
    batch = test_ds.sample_batch(count, total_len)  # 和 Prefetcher 里调用的方式一致
    return batch[token_key].to(device).long()       # 取 x，放到 GPU，转 long


@torch.no_grad()
def write_with_teacher_rnn(rnn, teacher, x, read_len, write_len, temp):
    x_read = x[:, :read_len]  # 提示部分，(n, read_len)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        # 初始记忆由 teacher 一次性编码满 256 字节得到，和 DMT 训练时的起点一致
        z = teacher.encode(x_read)  # (n, Z)
        x_write = []
        for step in range(write_len):
            logits = rnn.rnn_readout(z).float()  # 从记忆预测下一个 token 的分数，(n, V)
            if temp > 0:
                probs = torch.softmax(logits / temp, dim=-1)
                token = torch.multinomial(probs, num_samples=1)  # 按概率采样，(n, 1)
            else:
                token = logits.argmax(dim=-1, keepdim=True)      # 取最高分，(n, 1)
            x_write.append(token)
            z = rnn.rnn_update(z, token)  # 把新 token 写进记忆，得到下一步的 z
    return torch.cat(x_write, dim=1)      # (n, write_len)


def main():
    train_ds, test_ds, tokenizer = get_tinystories_dataset()
    rnn, teacher = load_models()

    total_len = prompt_len + gen_len
    ids = get_test_ids(test_ds, num_samples, total_len)
    prompt = ids[:, :prompt_len]               # 提示
    reference = ids[:, prompt_len:total_len]   # 真实后文，只用于对比
    new_tokens = write_with_teacher_rnn(rnn, teacher, ids, prompt_len, gen_len, temperature)

    for i in range(ids.size(0)):
        print(f"===== 样本 {i} =====")
        print("[提示结尾]", decode_text(tokenizer, prompt[i])[-prompt_show_chars:])
        print("[真实后文]", decode_text(tokenizer, reference[i]))
        print("[模型续写]", decode_text(tokenizer, new_tokens[i]))
        print()


if __name__ == "__main__":
    main()