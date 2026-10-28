
# pip install nvidia-ml-py psutil

# python monitor.py            # 默认 1 秒刷新
# python monitor.py -i 0.5     # 0.5 秒刷新


import argparse
import os
import time

import psutil
import pynvml

BAR_WIDTH = 30


def make_bar(percent):
    """把百分比转成定长文本进度条。"""
    percent = max(0.0, min(100.0, percent))
    filled = int(BAR_WIDTH * percent / 100)
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def format_line(label, percent, detail):
    return f"{label:<6} {make_bar(percent)} {percent:5.1f}%  {detail}"


def query_power(handle):
    """返回 (当前功耗W, 功耗上限W)，驱动不支持时返回 (None, None)。"""
    try:
        usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        return usage, limit
    except pynvml.NVMLError:
        return None, None


def build_lines(handles):
    lines = []

    cpu = psutil.cpu_percent(interval=None)
    lines.append(format_line("CPU", cpu, f"{psutil.cpu_count()} 逻辑核心"))

    for idx, handle in enumerate(handles):
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        power, power_limit = query_power(handle)

        lines.append("")
        lines.append(f"GPU{idx}: {name}")
        lines.append(format_line("GPU", util.gpu, ""))

        mem_pct = mem.used / mem.total * 100
        mem_detail = f"{mem.used / 1024 ** 3:.2f}/{mem.total / 1024 ** 3:.2f} GB"
        lines.append(format_line("VRAM", mem_pct, mem_detail))

        if power is None:
            lines.append(f"{'Power':<6} N/A")
        else:
            power_pct = power / power_limit * 100
            lines.append(format_line("Power", power_pct, f"{power:.1f}/{power_limit:.0f} W"))

    return lines


def main():
    parser = argparse.ArgumentParser(description="实时监控 GPU / CPU 使用情况")
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="刷新间隔(秒)，默认 1.0")
    args = parser.parse_args()

    # Windows 终端需要先执行一次，才会启用 ANSI 转义序列
    os.system("")

    pynvml.nvmlInit()
    handles = []
    for idx in range(pynvml.nvmlDeviceGetCount()):
        handles.append(pynvml.nvmlDeviceGetHandleByIndex(idx))

    # 预热：第一次调用 cpu_percent 会返回 0.0
    psutil.cpu_percent(interval=None)

    print("按 Ctrl+C 退出\n")
    last_height = 0

    try:
        while True:
            lines = build_lines(handles)

            # 光标上移到上一次输出的起点
            if last_height > 0:
                print(f"\033[{last_height}A", end="")
            for line in lines:
                # \033[K 清除当前行剩余内容，避免残留旧字符
                print(f"\033[K{line}")
            last_height = len(lines)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已退出")
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()

