
import torch

def normalize_mem(z, rms_norm=True, fsq=None):
    """
    Normalize the memory tokens with RMS norm and (optional) FSQ.
    """

    # 对最后一维（D，特征维度）做 RMSNorm：只做"除以均方根"缩放，不像 LayerNorm 那样减均值
    # z.shape = (B, T, D)
    # z.shape[-1:]  # (D,)
    # 传入 (D,)，表示 RMSNorm 只对最后一维 D（特征维度）进行归一化。
    # 例如，z 的形状为 (B, T, D)，那么每个记忆向量 z[b, t, :] 都会独立进行 RMSNorm，不会混合不同的 B 或 T 位置的数据。
    if rms_norm:
        z = torch.nn.functional.rms_norm(z, z.shape[-1:])

    return z

