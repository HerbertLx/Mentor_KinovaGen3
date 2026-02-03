import os
import sys

import torch
from torchvision.utils import save_image


def add_project_root_to_sys_path():
    """
    将项目根目录加入 sys.path，方便在测试脚本中导入 agents.mentor。
    当前文件结构假定为:
        Mentor_KinovaGen3/
            agents/mentor.py
            test_lx/RandomShiftsAug/test.py  (本文件)
    因此本文件的上上级目录即为项目根目录。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def test_random_shifts_aug():
    """
    对 agents.mentor.RandomShiftsAug 进行简单功能测试，并将前后对比图像保存到本目录。
    """
    from agents.mentor import RandomShiftsAug  # noqa: E402  (延迟导入以确保 sys.path 已更新)

    # 配置
    pad = 4
    batch_size = 8
    channels = 3
    height = width = 84  # 必须正方形，满足 assert h == w

    # 固定随机种子，方便复现
    torch.manual_seed(0)

    # 构造一批假图像数据: [n, c, h, w]，使用均匀分布更便于可视化
    x = torch.rand(batch_size, channels, height, width)

    aug = RandomShiftsAug(pad=pad)

    # 前向传播
    y = aug(x)

    # 1) 基本形状检查：输出形状应与输入完全一致
    assert y.shape == x.shape, f"输出形状 {y.shape} 与输入形状 {x.shape} 不一致"

    # 2) 数值有效性检查：不应包含 NaN 或 Inf
    assert torch.isfinite(y).all(), "输出包含 NaN 或 Inf"

    # 3) 随机性检查：
    #    - 在同一个对象上，两次前向传播结果一般应该不同（因为随机平移）
    y2 = aug(x)
    same_ratio = (y == y2).float().mean().item()
    print(f"相同像素比例（两次增强结果之间）：{same_ratio:.6f}")

    # 如果完全一样，说明随机性可能有问题；一般来说 same_ratio 应该远小于 1
    assert same_ratio < 1.0, "两次随机增强结果完全相同，可能随机性失效"

    # 4) 非正方形输入应触发断言
    non_square = torch.rand(batch_size, channels, height, width + 1)
    try:
        _ = aug(non_square)
    except AssertionError:
        print("非正方形输入正确地触发了断言。")
    else:
        raise AssertionError("非正方形输入未触发断言，行为与预期不符。")

    # 5) 保存部分样本的前后对比图像到当前目录
    output_dir = os.path.dirname(os.path.abspath(__file__))
    num_save = min(4, batch_size)
    for i in range(num_save):
        input_path = os.path.join(output_dir, f"input_{i}.png")
        aug_path = os.path.join(output_dir, f"aug_{i}.png")
        save_image(x[i], input_path)
        save_image(y[i], aug_path)
        print(f"已保存原图: {input_path}")
        print(f"已保存增强图: {aug_path}")

    print("RandomShiftsAug 基本测试通过，图像已保存到当前目录。")


if __name__ == "__main__":
    add_project_root_to_sys_path()
    test_random_shifts_aug()


