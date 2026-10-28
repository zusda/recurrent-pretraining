

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
class dmt_traning_config:
    seed = 0
    # dataset: str = "tinystories"

    # ------------------- nomal config -------------------
    batch_size = 32 # batch size
    n_iters = 1500 # number of optimization iterations   # 相比smt少了很多，应该也是自回归训练所需时间更长

    checkpoint_itr = 300  # save checkpoint every n iterations


    save_dir = "./runs/dmt_tinystories_0/"   # save directory

    smt_model_path = "./runs/smt_tinystories_0/"
    rnn_path = smt_model_path + "rnn.pt"
    teacher_path = smt_model_path +"teacher.pt"

    accum_steps = 2

    # ------------------- learning rate related -------------------
    lr = 3e-5 # learning rate   # 相比smt更小的学习率
    lr_schedule = "warmup_cosine"

    clip_grad_norm = 1 # clip gradient norm

    # ------------------- DMTConfig -------------------

    T_ctx: int = 256 # context length
    T_fut: int = 64 # future length
    Td: int = 256  # number of RNN rollout steps for DMT
    T_total: int = T_ctx+T_fut+Td # total sequence length


    # coef_dec: float = 1. # coefficient for the decoder predictive state loss.
    coef_rnn_enc: float = 10. # coefficient for the dynamics loss.
    coef_unif: float = 1e-2 # coefficient for the uniformity loss.
    coef_rnn_dec: float = 1. # coefficient for the readout loss.

    # 控制 DMT 训练时 rollout 过程中每隔多少步计算一次 loss
    k: int = 16 # subsampling factor for DMT rollout. If Td=16 and k=4, then loss is only calculated at t=0, 4, 8, 12.



class dmt_traning:
    def __init__(self, cfg: dmt_traning_config):
        self.cfg = cfg

        # 固定随机种子，保证可复现。
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        # 混合精度训练
        self.amp = torch.amp.autocast("cuda", dtype=torch.bfloat16) # mixed precision training

        # 加载数据集
        self.train_ds, self.test_ds, self.tokenizer = get_tinystories_dataset()
        T_total = self.cfg.T_total
        batch_batch = self.cfg.batch_size
        self.train_pf = tools.Prefetcher(self.train_ds, batch_batch, T_total)
        self.test_pf = tools.Prefetcher(self.test_ds, batch_batch, T_total, depth=2)


        # 初始化 RNN 和 Teacher
        self.rnn = RNN().cuda() # initialize the RNN
        self.teacher = Teacher().cuda() # initialize the teacher


        # 加载 smt 中训练好的模型
        if self.cfg.rnn_path is not None:
            self.rnn.load_state_dict(
                torch.load(
                    self.cfg.rnn_path,
                    map_location="cpu"
                    # map_location="cpu" 只决定 torch.load 读出来的那份权重临时放在哪，不影响模型。
                    # 模型已经在 GPU 上，load_state_dict 会把这份权重拷进模型已有的 GPU 参数里
                )
            )
        if self.cfg.teacher_path is not None:
            self.teacher.load_state_dict(torch.load(self.cfg.teacher_path, map_location="cpu"))


        # dmt中仅训练rnn
        parameters = []
        parameters += list(self.rnn.parameters())


        self.opt = torch.optim.AdamW(
            parameters,
            lr=self.cfg.lr,
            weight_decay=1e-2,
            betas=(0.9, 0.95))

        self.logs = []


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
        # 将所有待训练参数的梯度清空
        self.opt.zero_grad(set_to_none=True)


        # 计算梯度
        self.fw_bw_loss_calculation(logs, train_flag=True)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), self.cfg.clip_grad_norm)
        logs['grad_norm'] = grad_norm.item()
        logs['time'] = time.time()

        self.opt.step() # update parameters







    def run(self):
        # -------------------------------------------------------------------------
        # 断点续训：如果发现已有checkpoint，加载模型、优化器状态、日志、当前迭代数，从中断处继续；
        if (self.cfg.save_dir is not None and os.path.exists(f"{self.cfg.save_dir}/checkpoint.pt")
        ):
            self.restore_from_checkPoint()
            pbar = tqdm(range(self.itr, self.cfg.n_iters))
        else:
            pbar = tqdm(range(self.cfg.n_iters))
        # -------------------------------------------------------------------------


        for self.itr in pbar:
            self.step(self.itr)  # perform a single optimization step

            # 在 tqdm 进度条的末尾，实时显示当前的loss数值
            pbar.set_postfix(loss=f"{self.logs[-1]['loss']:.4f}")


            if self.cfg.checkpoint_itr is not None and (self.itr + 1) % self.cfg.checkpoint_itr == 0:
                self.save_to_checkPoint()


    def restore_from_checkPoint(self):
        """从 save_dir/checkpoint.pt 恢复 rnn、teacher、优化器、日志和迭代数。"""
        ckpt = torch.load(f"{self.cfg.save_dir}/checkpoint.pt", weights_only=False)
        self.rnn.load_state_dict(ckpt["rnn"])
        self.teacher.load_state_dict(ckpt["teacher"])
        self.opt.load_state_dict(ckpt["opt"])
        self.logs = ckpt["logs"]
        self.itr = ckpt["itr"]

    def save_to_checkPoint(self):
        """保存完整断点（rnn、teacher、优化器、日志、迭代数、配置）到 save_dir/checkpoint.pt，每次覆盖。"""
        # 保存前把 teacher 的读出层权重复制到 rnn，保证 checkpoint 里的 rnn 是完整的
        # self.rnn.decoder.load_state_dict(self.teacher.decoder.state_dict())
        # self.rnn.to_logits.load_state_dict(self.teacher.to_logits.state_dict())

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

    def split_batch(self,batch, parts):
        chunks = {k: v.chunk(parts, dim=0) for k, v in batch.items()}
        return [{k: chunks[k][i] for k in chunks} for i in range(parts)]

    def fw_bw_loss_calculation(self, logs, train_flag=False):
        # 计算梯度
        # 1. 取数据
        # 2. 前向传播，计算loss
        # 3. 反向传播
        outputs = []
        if train_flag:
            # batch = self.train_pf.get()
            # with self.amp:
            #     o = self.forward_dmt(self.rnn, self.teacher, batch)
            # o['loss'].backward()  # backward pass
            #
            # outputs.append({k: v.detach() for k, v in o.items() if v.numel() == 1})
            # outputs = {k: torch.stack([o[k] for o in outputs]).mean().item() for k in outputs[0].keys()}
            # logs.update({f"{k}": v for k, v in outputs.items()})

            accumulate = 1
            batch = self.train_pf.get()
            total = batch['x'].shape[0]  # 'x' 换成你的真实键名
            outputs = []
            for micro_batch in self.split_batch(batch, accumulate):
                n = micro_batch['x'].shape[0]  # 这一份的样本数
                with self.amp:
                    o = self.forward_dmt(self.rnn, self.teacher, micro_batch)
                (o['loss'] * n / total).backward()
                outputs.append({k: v.detach() * n / total for k, v in o.items() if v.numel() == 1})

            outputs = {k: torch.stack([o[k] for o in outputs]).sum().item() for k in outputs[0].keys()}
            logs.update({f"{k}": v for k, v in outputs.items()})



        else:
            batch = self.test_pf.get()
            with torch.no_grad(), self.amp:
                # torch.amp.autocast 主要针对GPU设计和优化，代码里传入的 "cuda" 这个参数，就是显式指定"这是给CUDA设备用的autocast"
                o = self.forward_dmt(self.rnn, self.teacher, batch)  # forward pass with SMT

            # 只收集那些numel()==1（标量）的输出项
            outputs.append({k: v.detach() for k, v in o.items() if v.numel() == 1})

            # 多个batch中, 相同Key的内容取平均值 (这里只有一个batch，不过复用smt代码懒得动了，反正效果一样)
            outputs = {k: torch.stack([o[k] for o in outputs]).mean().item() for k in outputs[0].keys()}
            logs.update({f"{k}_test": v for k, v in outputs.items()})



    def forward_dmt(self,rnn, teacher, batch):
        cfg = self.cfg
        x, y = batch['x'], batch['y']
        B, T = x.shape

        # loss_decoder = []
        z_enc = []

        # 用 teacher encoder 生成"标准答案"轨迹 z_enc
        # disable=True 不显示进度条
        for t in tqdm(range(0, cfg.Td, cfg.k), disable=True):
            x_ctx, x_fut, y_fut = tools.get_dmt_batch(batch, t, cfg)

            with torch.no_grad():
                z = teacher.encode(x_ctx) # 返回的是 (B, Z) —— Z经是"K个mem token展平后"的总维度
            z_enc.append(z)
            #   每个z是(B,Z)的形状，添加完后，z_enc是一个长度为Td的列表

        # torch.stack(z_enc, dim=1)：把长度Td的list，沿着新的第1维堆叠起来：z_enc.shape = (B, Td, Z)
        z_enc = torch.stack(z_enc, dim=1)



        # rnn enc 的输入z 是自己上一步enc输出的z。
        # dec
        loss_readout = []
        z_rnn = []
        z_t = z_enc[:, 0] # 初始化z_t，注意这里用的是z_enc[:,0]（encoder算出的第一个记忆,td=0的时候算出的m），不是从零初始化

        for t in tqdm(range(cfg.Td), disable=True):

            # 当前记忆z_t
            z_rnn.append(z_t)
            logits = rnn.rnn_readout(z_t.detach()) # 取"最后一个memory位置"的Decoder输出  去预测紧跟在 mem 后面的那一个token(x[:, t + cfg.T_ctx])

            loss_readout_t = tools.calc_cross_entropy_loss(logits, x[:, t + cfg.T_ctx])
            # y[:, t + cfg.T_ctx - 1] , 原作者用的这个，y本身是在decoder中对应输出位置的正确target，而这里mem和y没有任何对应关系，建议直接用x[:, t + cfg.T_ctx]

            loss_readout.append(loss_readout_t)

            # dmt.detach_rnn=False，使用 r_enc(z, 下一token) 更新记忆z
            z_t = rnn.rnn_update(z_t.detach(), x[:, [cfg.T_ctx + t]])


        # 堆叠成(B, Td, Z), 然后用[::cfg.k]每隔k步取一个，让z_rnn的时间步数量和z_enc对齐（因为z_enc本来就是每隔k步才算一次）。
        # [:, ::cfg.k] 是对第 1 维(时间维)做步长为 k 的下采样,每隔 k 步取一个。
        z_rnn = torch.stack(z_rnn, dim=1)[:, ::cfg.k]    # (B, ceil(Td / k), Z)


        # 汇总readout loss。
        loss_readout_ = torch.stack(loss_readout, dim=1)  # (B T)
        loss_readout = loss_readout_.mean()


        # 比较"RNN自己滚动生成的轨迹z_rnn" 和 "encoder生成的标准轨迹z_enc"之间的MSE
        loss_rnn_enc = F.mse_loss(z_rnn, z_enc) # (B T Z)


        # 和之前一样的均匀性正则，作用在encoder的输出上。
        loss_unif = tools.calc_uniformity_loss(z_enc)

        loss =  cfg.coef_rnn_enc*loss_rnn_enc + cfg.coef_unif*loss_unif + cfg.coef_rnn_dec*loss_readout
        # loss加权求和


        return dict(loss_readout=loss_readout, loss_rnn_enc=loss_rnn_enc, loss_unif=loss_unif, loss=loss)


if __name__ == "__main__":
    cfg = dmt_traning_config()
    main = dmt_traning(cfg)
    main.run()





