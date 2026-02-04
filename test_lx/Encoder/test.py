import os
import sys

import torch


def add_project_root_to_sys_path():
    """
    将项目根目录加入 sys.path，方便在测试脚本中导入 agents.mentor。
    当前文件结构假定为:
        Mentor_KinovaGen3/
            agents/mentor.py
            test_lx/Encoder/test.py  (本文件)
    因此本文件的上上级目录即为项目根目录。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def test_encoder_scratch_and_spawnnet():
    """
    对 agents.mentor.Encoder 进行简单形状测试：
    - 测试 'scratch' 模式下的输入/输出形状
    - 测试 'spawnnet' 模式下的输入/输出形状
    并将结果打印到终端、同时写入当前目录下的 summary.txt 文件。
    """
    from agents.mentor import Encoder  # noqa: E402  (延迟导入以确保 sys.path 已更新)

    device = torch.device("cpu")
    batch_size = 4
    height = width = 84

    lines = []

    # ===========================
    # 1) scratch 模式测试
    # ===========================
    obs_shape_scratch = (3, height, width)
    encoder_scratch = Encoder(
        obs_shape=obs_shape_scratch,
        encoder_type="scratch",
        resnet_fix=False,
        pretrained_factor=1.0,
    ).to(device)

    # 构造假输入，模拟 [0, 255] 区间的图像像素
    x_scratch = torch.rand(batch_size, *obs_shape_scratch) * 255.0
    x_scratch = x_scratch.to(device)

    with torch.no_grad():
        out_scratch = encoder_scratch(x_scratch)

    info_scratch = [
        "=== Encoder scratch 模式 ===",
        f"obs_shape        : {obs_shape_scratch}",
        f"input_tensor.shape: {tuple(x_scratch.shape)}",
        f"repr_dim         : {encoder_scratch.repr_dim}",
        f"output_tensor.shape: {tuple(out_scratch.shape)}",
        "",
    ]
    for l in info_scratch:
        print(l)
    lines.extend(info_scratch)

    # ===========================
    # 2) spawnnet 模式测试
    # ===========================
    # 假设有 2 个相机、3 个时间步 => 通道数 = 3 * n_camera * time_steps = 18
    n_camera = 2
    time_steps = 3
    channels_spawnnet = 3 * n_camera * time_steps
    obs_shape_spawnnet = (channels_spawnnet, height, width)

    encoder_spawnnet = Encoder(
        obs_shape=obs_shape_spawnnet,
        encoder_type="spawnnet",
        resnet_fix=True,        # 通常冻结预训练 ResNet 以减少测试时的梯度计算
        pretrained_factor=0.5,  # 随便选择一个合理的缩放因子
    ).to(device)

    x_spawnnet = torch.rand(batch_size, *obs_shape_spawnnet) * 255.0
    x_spawnnet = x_spawnnet.to(device)

    with torch.no_grad():
        out_spawnnet = encoder_spawnnet(x_spawnnet)

    info_spawnnet = [
        "=== Encoder spawnnet 模式 ===",
        f"obs_shape        : {obs_shape_spawnnet}",
        f"input_tensor.shape: {tuple(x_spawnnet.shape)}",
        f"repr_dim         : {encoder_spawnnet.repr_dim}",
        f"output_tensor.shape: {tuple(out_spawnnet.shape)}",
        "",
    ]
    for l in info_spawnnet:
        print(l)
    lines.extend(info_spawnnet)

    # ===========================
    # 3) 将结果写入当前目录下的 summary.txt
    # ===========================
    output_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

    print(f"测试结果已写入: {summary_path}")


if __name__ == "__main__":
    add_project_root_to_sys_path()
    test_encoder_scratch_and_spawnnet()


