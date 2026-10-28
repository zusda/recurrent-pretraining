

# import os
# os.environ["PYTHONUTF8"] = "1"

import os
import torch
from dataclasses import dataclass, field
import numpy as np
import tyro
from tqdm.auto import tqdm

import pandas as pd
import time


from data.data_wrapper import get_tinystories_dataset
from model.teacher import  Teacher
from model.RNN import RNN
import training.tools as tools
import json
from einops import rearrange
import torch.nn.functional as F

@dataclass
class smt_traning_config:
    seed = 0
    # dataset: str = "tinystories"

    # ------------------- nomal config -------------------
    batch_size = 128 # batch size
    n_iters = 150000 # number of optimization iterations
    # log_every = 5000 # log every n iterations

    checkpoint_itr = 1000  # save checkpoint every n iterations

    save_dir = "./runs/smt_tinystories_0/"   # save directory



    # ------------------- learning rate related -------------------
    lr = 3e-4 # learning rate
    lr_schedule = "warmup_constant"

    grad_accum_steps = 1 # gradient accumulation steps
    clip_grad_norm = 1 # clip gradient norm

    # ------------------- SMTConfig -------------------
    T_ctx: int = 256 # context length
    T_fut: int = 64 # future length
    T_total: int = T_ctx+T_fut # total sequence length

    gamma: float = 1.  # teacher decoder中给不同x_fut时间步的loss加的指数衰减权重，这里为1，说明不衰减。

    coef_dec = 1.           # coefficient for teacher decoder.
    coef_rnn_enc = 0.1          # coefficient for rnn encoder.
    coef_unif = 0.001       # coefficient for the uniformity loss.



class smt_traning:
    def __init__(self, cfg: smt_traning_config):
        self.cfg = cfg

        # 保证batch能被"梯度累积步数"整除（每次实际forward的子batch大小要能整除）
        assert cfg.batch_size % cfg.grad_accum_steps == 0

        # 固定随机种子，保证可复现。
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        # 混合精度训练：训练时不全部用 fp32（32 位浮点），而是让大部分计算用 16 位浮点（fp16 或 bf16），只在对精度敏感的地方保留 fp32。
        # （PyTorch 默认是 fp32）
        # 同时bfloat16比float16数值范围更大，训练更稳定（这是现代大模型训练的标配技巧）。
        self.amp = torch.amp.autocast("cuda", dtype=torch.bfloat16) # mixed precision training


        # 加载数据集
        self.train_ds, self.test_ds, self.tokenizer = get_tinystories_dataset()
        T_total = self.cfg.T_total
        micro_batch = self.cfg.batch_size // self.cfg.grad_accum_steps  # micro-batch 大小
        self.train_pf = tools.Prefetcher(self.train_ds, micro_batch, T_total)
        self.test_pf = tools.Prefetcher(self.test_ds, micro_batch, T_total, depth=2)


        # 初始化 RNN 和 Teacher
        self.rnn = RNN().cuda() # initialize the RNN
        self.teacher = Teacher().cuda() # initialize the teacher

        # self.rnn = torch.compile(self.rnn)
        # self.teacher = torch.compile(self.teacher)


        parameters = []
        parameters += list(self.rnn.parameters())
        parameters += list(self.teacher.parameters())

        self.opt = torch.optim.AdamW(
            parameters,
            lr=self.cfg.lr,
            weight_decay=1e-2, # 权重衰减
            betas=(0.9, 0.95)   # 基本不用动。LLM 训练的事实标准（GPT-3、LLaMA、nanoGPT 都用）
        )

        self.logs = []





    # 1 个 batch 的训练
    def step(self, itr=0):

        # 初始化这一步的日志字典。
        logs = dict(itr=itr)
        self.logs.append(logs)

        # --------------------------------------------------------
        # 每10轮，计算一次test上的loss值,记录在log中
        if itr % 10 == 0:
            self.rnn.eval(); self.teacher.eval()
            self.fw_bw_loss_calculation(logs, train_flag=False)
        # --------------------------------------------------------


        # 获取学习率r
        lr = tools.get_lr(itr, self.cfg.n_iters, self.cfg.lr, self.cfg.lr_schedule)  # warmup_constant
        # 我们 torch.optim.AdamW(parameters,...) , 因此 param_groups只有一个group
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr
        logs['lr'] = lr

        # 设置 模型为train模式
        self.rnn.train(); self.teacher.train()
        # set_to_none=True 比设置全0张量各方面都好。较新版本的PyTorch，zero_grad()的默认值本身也已经是True了
        self.opt.zero_grad(set_to_none=True)


        # 计算梯度
        self.fw_bw_loss_calculation(logs, train_flag=True)

        # 裁剪更新梯度
        grad_norm_rnn = torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), self.cfg.clip_grad_norm)
        grad_norm_teacher = torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), self.cfg.clip_grad_norm)
        self.opt.step() # update parameters

        logs['grad_norm_rnn'] = grad_norm_rnn.item()
        logs['grad_norm_teacher'] = grad_norm_teacher.item()
        logs['time'] = time.time()


    # 前向传播计算loss,以及反向传播，的整个过程
    def fw_bw_loss_calculation(self, logs, train_flag=False):
        # 计算梯度
        # 1. 取数据
        # 2. 前向传播，计算loss
        # 3. 反向传播
        outputs = []
        if train_flag:
            for _ in range(self.cfg.grad_accum_steps):
                batch = self.train_pf.get()

                # 前向传播
                with self.amp:
                    o = self.forward_smt(self.rnn, self.teacher, batch)  # forward pass with SMT

                # 反向传播，计算梯度
                (o['loss'] / self.cfg.grad_accum_steps).backward()  # backward pass


                # 只收集那些numel()==1（标量）的输出项
                outputs.append({k: v.detach() for k, v in o.items() if v.numel() == 1})

            # 多个batch中, 相同Key的内容取平均值
            outputs = {k: torch.stack([o[k] for o in outputs]).mean().item() for k in outputs[0].keys()}
            logs.update({f"{k}": v for k, v in outputs.items()})


        else:

            for _ in range(self.cfg.grad_accum_steps):
                batch = self.test_pf.get()

                with torch.no_grad(), self.amp:
                    # torch.amp.autocast 主要针对GPU设计和优化，代码里传入的 "cuda" 这个参数，就是显式指定"这是给CUDA设备用的autocast"
                    o = self.forward_smt(self.rnn, self.teacher, batch)  # forward pass with SMT

                # 只收集那些numel()==1（标量）的输出项
                outputs.append({k: v.detach() for k, v in o.items() if v.numel() == 1})

            # 多个batch中, 相同Key的内容取平均值
            outputs = {k: torch.stack([o[k] for o in outputs]).mean().item() for k in outputs[0].keys()}
            logs.update({f"{k}_test": v for k, v in outputs.items()})





    # 整个训练主循环
    def run(self):
        # 断点续训：如果发现已有checkpoint，加载模型、优化器状态、日志、当前迭代数，从中断处继续；
        if (self.cfg.save_dir is not None and os.path.exists(f"{self.cfg.save_dir}/checkpoint.pt")
        ):
            self.restore_from_checkPoint()
            pbar = tqdm(range(self.itr, self.cfg.n_iters))
        else:
            pbar = tqdm(range(self.cfg.n_iters))


        for self.itr in pbar:
            # 主循环，每次调用step。 内部进行 前向/反向 传播，loss计算与参数更新。
            self.step(self.itr) # perform a single optimization step

            # 在 tqdm 进度条的末尾，实时显示当前的loss数值
            pbar.set_postfix(loss=f"{self.logs[-1]['loss']:.4f}")

            # print(self.logs)
            # input('test 中，end')
            if self.cfg.checkpoint_itr is not None and (self.itr + 1) % self.cfg.checkpoint_itr == 0:
                self.save_to_checkPoint()



    def restore_from_checkPoint(self):
        """从 save_dir/checkpoint.pt 恢复 rnn、teacher、优化器、日志和迭代数。"""
        print('restore from checkpoint.pt')
        ckpt = torch.load(f"{self.cfg.save_dir}/checkpoint.pt", weights_only=False)
        self.rnn.load_state_dict(ckpt["rnn"])
        self.teacher.load_state_dict(ckpt["teacher"])
        self.opt.load_state_dict(ckpt["opt"])
        self.logs = ckpt["logs"]
        self.itr = ckpt["itr"]

    def save_to_checkPoint(self):
        """保存完整断点（rnn、teacher、优化器、日志、迭代数、配置）到 save_dir/checkpoint.pt，每次覆盖。"""
        # 保存前把 teacher 的读出层权重复制到 rnn，保证 checkpoint 里的 rnn 是完整的

        # smt中，rnn和teacher的decoder的模型是一样的，我们不训练rnn decoder
        # 我们可以self.rnn.decoder.load_state_dict(self.teacher.decoder.state_dict())，把teacher的decoder参数给rnn的decoder，当作dmt时候的初始参数。
        self.rnn.decoder.load_state_dict(self.teacher.decoder.state_dict())
        # 最后一层的线性层也同理
        self.rnn.to_logits.load_state_dict(self.teacher.to_logits.state_dict())


        os.makedirs(self.cfg.save_dir, exist_ok=True)
        torch.save({
            "cfg": self.cfg,
            "logs": self.logs,
            "itr": self.itr + 1,  # 已完成的迭代数，也是续训时的起始 itr
            "rnn": self.rnn.state_dict(),
            "teacher": self.teacher.state_dict(),
            "opt": self.opt.state_dict(),
        }, os.path.join(self.cfg.save_dir, "checkpoint.pt"))

        # 2. 日志
        logs_path = os.path.join(self.cfg.save_dir, f"logs.json")
        with open(logs_path, "w") as f:
            json.dump(self.logs, f)


        torch.save(self.rnn.state_dict(), os.path.join(self.cfg.save_dir, "rnn.pt"))
        torch.save(self.teacher.state_dict(), os.path.join(self.cfg.save_dir, "teacher.pt"))



    def forward_smt(self,rnn, teacher, batch):
        cfg = self.cfg
        # x, y = batch['x'], batch['y']

        x_ctx, x_fut, y_fut, x_ctx_next = tools.get_smt_batch(batch, cfg.T_ctx, cfg.T_fut)

        # ----------------------------------------------------------------------
        # 训练 teacher 的 encoder模型，使它包含能够预测 x_fut 内容的信息
        z_now = teacher.encode(x_ctx)
        logits = teacher.decode(z_now, x_fut)  # (B, Tf, V)
        losses_decoder = tools.calc_cross_entropy_loss(logits, y_fut.clamp(min=0))


        # 给不同时间步的loss加了指数衰减权重，这里 gamma=1，不衰减
        # loss_decoder_decayed = losses_decoder * (cfg.gamma ** torch.arange(cfg.T_fut, device=x_ctx.device))

        # batch数据无padding内容，不需要像原作者代码中用mask排除padding位置的loss
        loss_decoder = losses_decoder.mean()
        # ----------------------------------------------------------------------



        # ----------------------------------------------------------------------
        # （正则项，防止记忆塌缩）用来防止表示塌缩（即防止模型为了"作弊"降低其他loss，把所有样本的记忆都学成几乎相同的向量）
        loss_unif = tools.calc_uniformity_loss(rearrange(z_now, "B (K d) -> B K d", K=teacher.cfg.mem_vecs))
        # ----------------------------------------------------------------------



        # ----------------------------------------------------------------------
        # 训练rnn的 encoder 模型，使它能够做到 z= enc( z , x_next )
        z_next = teacher.encode(x_ctx_next)

        # 只取x_fut的第一个token，用[0]保持维度是(B,1)。这是准备喂给 rnn.rnn_update 用的。只需要"context之后紧跟着的下一个真实token"
        x_in = x_fut[:, [0]]  # (B, 1)

        # mem_next = f(mem,x)
        z_next_pred = rnn.rnn_update(z_now, x_in)

        loss_rnn_enc = F.mse_loss(z_next_pred,  z_next) # dynamics loss
        # ----------------------------------------------------------------------

        #     coef_dec: float = 1. # coefficient for the decoder predictive state loss.
        #     coef_dyn: float = 0.1 # coefficient for the dynamics loss.
        #     coef_unif: float = 0.001 # coefficient for the uniformity loss.
        # 三项loss按各自系数加权求和
        loss = cfg.coef_dec*loss_decoder + cfg.coef_rnn_enc*loss_rnn_enc + cfg.coef_unif*loss_unif # total loss

        return dict(loss_decoder=loss_decoder, loss_rnn_enc=loss_rnn_enc, loss_unif=loss_unif, loss=loss)


if __name__ == "__main__":
    cfg = smt_traning_config()
    main = smt_traning(cfg)
    main.run()





