"""
SMT 阶段模型的评测脚本，放在 training/test_smt.py。

做四件事（和作者 notebook 的评测方式对齐：初始记忆由 teacher 编码满 256 字节得到）：
1. 一步动力学误差：teacher 状态 z_t 经 rnn_update 一步，和 teacher 的 z_{t+1} 的 MSE（对应 SMT 的 loss_dyn）。
2. 漂移曲线：RNN 从 z_enc[0] 自己滚 DRIFT_LEN 步，每隔 DRIFT_K 步和 teacher 状态比 MSE（对应 DMT 的 loss_dyn）。
3. 逐位置读出 loss：RNN 与 teacher 在同一批真实文本上、同一位置的下一字节交叉熵。
4. 自回归生成：teacher 生成 vs RNN 生成（初始记忆都来自 teacher.encode）。

想测 DMT 的结果，把 MODEL_DIR 改成 runs/dmt_tinystories_0 即可。
"""
import sys
from pathlib import Path

# training/ 的上一级就是项目根目录；加进 sys.path 后，无论从哪里启动都能 import data / model
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm.auto import tqdm

from data.data_wrapper import get_tinystories_dataset
from model.RNN import RNN
from model.teacher import Teacher

# ------------------- 配置 -------------------
MODEL_DIR = ROOT / "runs" / "smt_tinystories_0"
RNN_PATH = MODEL_DIR / "rnn.pt"
TEACHER_PATH = MODEL_DIR / "teacher.pt"

SEED = 0
BATCH = 16          # 评测用的样本条数
T_CTX = 256         # 上下文长度，必须和训练一致
EVAL_LEN = 128      # 逐位置 loss 的评测长度
DRIFT_LEN = 256     # 漂移曲线的 rollout 长度（对应 DMT 的 Td）
DRIFT_K = 16        # 漂移曲线每隔多少步取一个点（对应 DMT 的 k）
WRITE_LEN = 200     # 生成长度
TEMPERATURE = 0.8   # <=0 表示 greedy
NUM_SHOW = 3        # 打印几条生成样本
ONE_STEP_OFFSETS = [0, 16, 32, 48]


def teacher_readout(teacher, z, mem_vecs):
    """只用记忆 z 预测下一个字节。返回 logits，形状 (B, V)。"""
    z = rearrange(z, "B (K d) -> B K d", K=mem_vecs)
    return teacher.to_logits(teacher.decoder(z)[:, -1, :])


def sample_token(logits, temperature):
    """从 logits (B, V) 采样一个 token，返回 (B, 1)。"""
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def decode_tokens(tokenizer, tokens):
    """tokens: (B, T) 的整数张量，返回长度为 B 的字符串列表。"""
    arr = tokens.cpu().numpy()
    try:
        out = tokenizer.decode(arr)
        if isinstance(out, (list, tuple)) and len(out) == len(arr):
            return list(out)
    except Exception:
        pass
    # 字节级词表：token id 就是字节值，直接解码
    return [bytes(row.tolist()).decode("utf-8", errors="replace") for row in arr]


def one_step_mse(rnn, teacher, x, offsets):
    """teacher 状态走一步 rnn_update，与 teacher 的下一个状态比 MSE（对应 SMT 的 loss_dyn），对多个起点取平均。"""
    values = []
    for s in offsets:
        z_now = teacher.encode(x[:, s:s + T_CTX])
        z_next = teacher.encode(x[:, s + 1:s + 1 + T_CTX])
        z_pred = rnn.rnn_update(z_now, x[:, [s + T_CTX]])
        values.append(F.mse_loss(z_pred.float(), z_next.float()).item())
    return float(np.mean(values))


def drift_curve(rnn, teacher, x, drift_len, k):
    """
    RNN 从 teacher 编码的初始状态出发，用真实 token 自己滚 drift_len 步。
    每隔 k 步，和 teacher 在同一窗口的状态比 MSE。
    返回 (steps, mse_values, baseline)，baseline 是"两个无关样本的状态"之间的 MSE，用来衡量 MSE 的量级。
    """
    z_enc_list = []
    for t in range(0, drift_len, k):
        z_enc_list.append(teacher.encode(x[:, t:t + T_CTX]))
    z_enc = torch.stack(z_enc_list, dim=1).float()  # (B, ceil(drift_len / k), Z)

    z = teacher.encode(x[:, :T_CTX])
    steps = []
    values = []
    for t in range(drift_len):
        if t % k == 0:
            steps.append(t)
            values.append(F.mse_loss(z.float(), z_enc[:, t // k]).item())
        z = rnn.rnn_update(z, x[:, [T_CTX + t]])

    baseline = F.mse_loss(z_enc, z_enc.roll(1, dims=0)).item()
    return steps, values, baseline


def read_loss_curves(rnn, teacher, x, eval_len, mem_vecs):
    """
    逐位置下一字节交叉熵（真实 token 喂入，不是自回归生成）。
    返回两条长度为 eval_len 的曲线：RNN 的，teacher 的。
    位置 t 预测的是 x[:, T_CTX + t]。
    """
    z_rnn = teacher.encode(x[:, :T_CTX])
    loss_rnn = []
    loss_teacher = []
    for t in tqdm(range(eval_len), desc="read loss"):
        target = x[:, T_CTX + t]

        logits_rnn = rnn.rnn_readout(z_rnn)
        loss_rnn.append(F.cross_entropy(logits_rnn.float(), target, reduction="none"))

        z_enc = teacher.encode(x[:, t:t + T_CTX])
        logits_teacher = teacher_readout(teacher, z_enc, mem_vecs)
        loss_teacher.append(F.cross_entropy(logits_teacher.float(), target, reduction="none"))

        z_rnn = rnn.rnn_update(z_rnn, x[:, [T_CTX + t]])

    loss_rnn = torch.stack(loss_rnn, dim=1).mean(dim=0)
    loss_teacher = torch.stack(loss_teacher, dim=1).mean(dim=0)
    return loss_rnn.cpu().numpy(), loss_teacher.cpu().numpy()


def write_with_teacher(teacher, x_read, write_len, temperature, mem_vecs):
    """teacher 自回归生成：每生成一个字节，就把最近 T_CTX 个字节重新编码一次。返回 (B, write_len)。"""
    tokens = x_read
    generated = []
    z = teacher.encode(tokens[:, -T_CTX:])
    for step in tqdm(range(write_len), desc="teacher write"):
        token = sample_token(teacher_readout(teacher, z, mem_vecs), temperature)
        generated.append(token)
        tokens = torch.cat([tokens, token], dim=1)
        z = teacher.encode(tokens[:, -T_CTX:])
    return torch.cat(generated, dim=1)


def write_with_rnn(rnn, teacher, x_read, write_len, temperature):
    """初始记忆来自 teacher.encode，之后只用 RNN 自回归生成。返回 (B, write_len)。"""
    z = teacher.encode(x_read[:, -T_CTX:])
    generated = []
    for step in tqdm(range(write_len), desc="rnn write"):
        token = sample_token(rnn.rnn_readout(z), temperature)
        generated.append(token)
        z = rnn.rnn_update(z, token)
    return torch.cat(generated, dim=1)


def flat(text):
    """把换行显示成 \\n，方便一行看完。"""
    return text.replace("\n", "\\n")


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    amp = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    train_ds, test_ds, tokenizer = get_tinystories_dataset()

    rnn = RNN().cuda()
    teacher = Teacher().cuda()
    rnn.load_state_dict(torch.load(RNN_PATH, map_location="cpu"))
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location="cpu"))
    rnn.eval()
    teacher.eval()
    mem_vecs = rnn.cfg.mem_vecs
    print("加载模型:", MODEL_DIR)

    total_len = T_CTX + max(EVAL_LEN, DRIFT_LEN)
    batch = test_ds.sample_batch(BATCH, total_len)
    x = batch["x"].cuda().long()
    print("评测数据形状:", tuple(x.shape))

    with torch.no_grad(), amp:
        # 1. 一步动力学误差
        dyn = one_step_mse(rnn, teacher, x, ONE_STEP_OFFSETS)
        print("\n===== 1. 一步动力学 MSE（应接近 SMT 日志里的 loss_dyn_test）=====")
        print(f"one-step mse = {dyn:.6f}")

        # 2. 漂移曲线
        steps, values, baseline = drift_curve(rnn, teacher, x, DRIFT_LEN, DRIFT_K)
        print("\n===== 2. 漂移曲线：RNN 自己滚 vs teacher 状态 =====")
        print(f"无关样本之间的 MSE 基线 = {baseline:.4f}")
        for step, value in zip(steps, values):
            print(f"  step {step:4d}  mse = {value:.6f}")

        # 3. 逐位置读出 loss
        loss_rnn, loss_teacher = read_loss_curves(rnn, teacher, x, EVAL_LEN, mem_vecs)
        print("\n===== 3. 逐位置下一字节 loss（RNN vs teacher）=====")
        print("位置 0 时 RNN 的状态就是 teacher 的状态，两者应当相同；不同说明 rnn 的读出头不是 teacher 的")
        positions = [0, 1, 2, 4, 8, 16, 32, 64, 96, EVAL_LEN - 1]
        shown = []
        for p in positions:
            if p < EVAL_LEN and p not in shown:
                shown.append(p)
                print(f"  pos {p:4d}  rnn = {loss_rnn[p]:.4f}  teacher = {loss_teacher[p]:.4f}")
        print(f"  平均      rnn = {loss_rnn.mean():.4f}  teacher = {loss_teacher.mean():.4f}")

        # 4. 自回归生成
        x_read = x[:, :T_CTX]
        out_teacher = write_with_teacher(teacher, x_read, WRITE_LEN, TEMPERATURE, mem_vecs)
        out_rnn = write_with_rnn(rnn, teacher, x_read, WRITE_LEN, TEMPERATURE)

    prompts = decode_tokens(tokenizer, x_read)
    reals = decode_tokens(tokenizer, x[:, T_CTX:T_CTX + WRITE_LEN])
    texts_teacher = decode_tokens(tokenizer, out_teacher)
    texts_rnn = decode_tokens(tokenizer, out_rnn)

    print("\n===== 4. 生成样本（temperature =", TEMPERATURE, "）=====")
    for i in range(NUM_SHOW):
        print(f"\n--- 样本 {i} ---")
        print("[提示结尾]   ", flat(prompts[i][-100:]))
        print("[真实后文]   ", flat(reals[i]))
        print("[teacher续写]", flat(texts_teacher[i]))
        print("[rnn续写]    ", flat(texts_rnn[i]))


if __name__ == "__main__":
    main()


'''
这份结果说明 SMT 阶段是健康的，而 RNN 在滚动十几步后彻底崩掉，正是 DMT 要修的问题，所以它本身不算异常。

逐项解读

teacher 没问题。 平均下一字节 loss 0.676（约 0.98 bit/字节），续写是通顺的英文，偶有拼写错误（dideals、snakens），在 temperature 0.8 下属于正常。

SMT 同时训练了两个东西，所以要看两项：

teacher 续写正常：说明 teacher 的 encoder 和 decoder 训好了，也就是"记忆状态里保存了足够预测未来的信息"。
一步动力学 MSE = 0.000454：说明 RNN 学会了 rnn_update，从 teacher 的状态走一步，几乎能精确落在 teacher 的下一个状态上（无关样本基线是 1.77）。

----------------------

F:\software\python3_12_6\python.exe F:\code\python\RNN_with_smt_dmt\training\test_smt.py 
加载模型: F:\code\python\RNN_with_smt_dmt\runs\smt_tinystories_0
评测数据形状: (16, 512)

===== 1. 一步动力学 MSE（应接近 SMT 日志里的 loss_dyn_test）=====
one-step mse = 0.000454

===== 2. 漂移曲线：RNN 自己滚 vs teacher 状态 =====
无关样本之间的 MSE 基线 = 1.7652
  step    0  mse = 0.000000
  step   16  mse = 0.287841
  step   32  mse = 1.811777
  step   48  mse = 1.843434
  step   64  mse = 1.930430
  step   80  mse = 1.929923
  step   96  mse = 1.930999
  step  112  mse = 1.927850
  step  128  mse = 1.931176
  step  144  mse = 1.930367
  step  160  mse = 1.929245
  step  176  mse = 1.931103
  step  192  mse = 1.931979
  step  208  mse = 1.931337
  step  224  mse = 1.931884
  step  240  mse = 1.931329
read loss: 100%|██████████| 128/128 [00:10<00:00, 12.30it/s]

===== 3. 逐位置下一字节 loss（RNN vs teacher）=====
位置 0 时 RNN 的状态就是 teacher 的状态，两者应当相同；不同说明 rnn 的读出头不是 teacher 的
  pos    0  rnn = 0.5547  teacher = 0.5547
  pos    1  rnn = 0.2340  teacher = 0.2290
  pos    2  rnn = 0.2771  teacher = 0.2809
  pos    4  rnn = 1.2717  teacher = 1.2163
  pos    8  rnn = 0.7131  teacher = 0.7051
  pos   16  rnn = 2.0992  teacher = 0.3998
  pos   32  rnn = 5.1257  teacher = 0.8491
  pos   64  rnn = 5.9391  teacher = 0.6072
  pos   96  rnn = 4.6757  teacher = 0.6873
  pos  127  rnn = 4.8522  teacher = 0.5571
  平均      rnn = 4.2294  teacher = 0.6761
teacher write: 100%|██████████| 200/200 [00:08<00:00, 23.66it/s]
rnn write: 100%|██████████| 200/200 [00:08<00:00, 24.62it/s]

===== 4. 生成样本（temperature = 0.8 ）=====

--- 样本 0 ---
[提示结尾]    t of the kitchen, holding hands and eating their cookies. They were happy.Tommy liked to whistle. He
[真实后文]     whistled all the time. He whistled at home, at school, and on the bus. He did not care if people told him to stop. He was stubborn.\n\nOne day, he whistled on the bus so loud that the driver got angry.
[teacher续写]  would whistle his toys and go under the big rocks. He whistled and sang on his bell. He whistled and smiled. He would whistle and make signs.\n\nOne day, Tommy saw a new bird on the tree. The bird was 
[rnn续写]      whistled "speecial, plane uauuuauu,e,uAuuaauuuauu,auauuuuuVu, euua,auaeuaaVuuuu!arVuuaauiua,u;uaaGuruVamauhiureaa auuauuuA,uuaasu_uIauG,aaaraa3aeauar,uaubuuuluuu 0uaiuurhhuuV,euAuuV;eu uauaaaurau

--- 样本 1 ---
[提示结尾]    alarm. \n\nShe took off her coat and changed her clothes. Then she went to the kitchen and made dinner
[真实后文]    . She ate yummy food and went to sleep. \n\nMommy had a nice night of sleep so she could do more work in the morning.Mia ran in the race. She was so excited but a little scared too. She saw lots of peop
[teacher续写]  with yummy candy. It was a special day for it! The little girl was so happy that she ran to sleep and feel safe.Mum and Dad Sackie were walking around the school. There was a big yellow and baby in a
[rnn续写]     . She was so happy anvvu~VAerVsVuelzeuu-aauu;V;ua;auYVmauu;Gu;aU;aueV>es,aaa,;VuVQV,Gueu;uu3earuu!ee;:r,;e!e!;uyuuV:u;uVua;aauVluel;ao;raaVVe;uemu/ueuaa,0auIea^snuu-ra9Vau,u,au;uuVGuG;u

--- 样本 2 ---
[提示结尾]     and made it round and smooth.\n\n"Look at my snake!" Mia said. "It can hiss and bite!"\n\n"Look at my b
[真实后文]    all!" Ben said. "It can bounce and roll!"\n\nThey showed each other their clay creations. They were proud of them.\n\nBut then, a big boy came. He was older and taller than Mia and Ben. He was ignorant. H
[teacher续写] lack claye!" Ben said. "I can pull it to pick him."\n\nThey argued of their dideals. They pulled it and started to make the black snake. They stepped to share their snakens and share their clay.\n\nBut th
[rnn续写]     all!" Ben said. "Yomu3YzuAlauuV,;Nauauu-,MaVaVuau;uVruu;;,Auu uaeuuafauuu;Vu0uia-eru,eVauumaur;a,r;u0GuuaauuAVz0!;Guuauueuaauiaauuuu,uGa;uuvuueTuue;uuua!:Vreu;ua,urVlUaluruVlouaGGuuuVGa;ua 

进程已结束，退出代码为 0

'''





