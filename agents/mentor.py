import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("..")
import utils

from moe import MoE
import numpy as np
import torchvision.models as models

class RandomShiftsAug(nn.Module):
    """
    随机平移数据增强类。
    
    主要功能:
        对输入的图像进行随机的小幅度平移。
        这种增强方式在基于视觉的强化学习（如 DrQ-v2）中非常有效，
        能显著提升智能体对位置变化的鲁棒性。
    """
    def __init__(self, pad):
        """
        初始化增强器。

        参数说明:
            pad (int): 填充的像素大小。例如 pad=4 表示图像可以上下左右最多移动 4 个像素。
        """
        super().__init__()
        self.pad = pad

    def forward(self, x):
        """
        前向传播逻辑：执行随机平移。

        输入参数:
            x (torch.Tensor): 输入图像张量，维度为 [n, c, h, w]。
        """
        n, c, h, w = x.size()
        # 内部逻辑: 这是一个约束，要求输入的图像必须是正方形
        assert h == w
        
        # 1. 图像填充 (Padding)
        # 将图像四周各填充 self.pad 个像素。'replicate' 模式会重复边界像素的值。
        # 填充后的维度变为 [n, c, h+2*pad, w+2*pad]
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        
        # 2. 生成基础网格 (Base Grid)
        # eps 是网格坐标的单位跨度
        eps = 1.0 / (h + 2 * self.pad)
        # 创建一个从 -1 到 1 的线性空间坐标，代表归一化后的像素位置
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]  # 只取前 h 个元素，为后续裁剪做准备
        
        # 将一维坐标扩展为二维坐标网格
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        # 此时 base_grid 的维度为 [1, h, w, 2]，包含了标准的采样坐标
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        # 3. 生成随机偏移量 (Random Shift)
        # 为 Batch 中的每张图片随机生成一个整数偏移量 [0, 2*pad]
        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        # 将像素单位的整数偏移转换为 [-1, 1] 范围内的浮点偏移量
        shift *= 2.0 / (h + 2 * self.pad)

        # 4. 合成最终采样网格
        # 基础网格 + 随机偏移 = 移动后的采样网格
        grid = base_grid + shift
        
        # 5. 空间重采样 (Spatial Resampling)
        # 使用 F.grid_sample 根据计算好的网格从填充后的图像 x 中采样。
        # 效果相当于在填充后的图像上随机切出了一个 h x w 的窗口。
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    """
    视觉编码器，将输入图像观测编码为一维特征向量。
    
    当前支持两种编码模式：
    - 'scratch':  使用从零开始训练的简单卷积网络。
    - 'spawnnet': 结合预训练 ResNet18 与自建卷积网络的“产卵网络”结构，用于多相机、多时间步输入。
    """
    
    def __init__(self, obs_shape, encoder_type, resnet_fix, pretrained_factor):
        """
        初始化编码器。
        
        参数说明:
            obs_shape (tuple[int, int, int]):
                观测张量的形状 (C, H, W)。
                - 对于 'scratch'：一般为单相机、单时间步的图像，例如 (3, 84, 84)。
                - 对于 'spawnnet'：一般为多相机、多时间步按通道拼接后的图像，
                  例如 3 * n_camera * time_steps 通道。
            encoder_type (str):
                编码器类型，必须为:
                - 'scratch'  : 纯自建卷积网络。
                - 'spawnnet' : 使用预训练 ResNet18 + 自建卷积网络的混合结构。
            resnet_fix (bool):
                仅在 'spawnnet' 模式下有效。
                - True  : 冻结预训练 ResNet18 的所有参数，仅训练自建卷积层和后续层。
                - False : 允许微调 ResNet18。
            pretrained_factor (float):
                仅在 'spawnnet' 模式下使用。
                用于缩放 ResNet 分支的特征，在与 scratch 分支特征拼接/融合时控制预训练特征的权重。
        
        属性:
            encoder_type (str): 当前编码器使用的模式。
            repr_dim (int): 编码后的特征向量维度，后续 Actor/Critic 等网络会依赖此值。
        """
        super().__init__()
        
        # 基本形状检查：观测必须为 (C, H, W) 三维
        assert len(obs_shape) == 3, f"obs_shape 必须为 (C, H, W)，但得到 {obs_shape}"
        
        # 保存编码器配置
        self.encoder_type = encoder_type  # 编码器类型：'scratch' 或 'spawnnet'
        self.pretrained_factor = pretrained_factor  # 预训练分支特征缩放因子
        
        # 确保传入的 encoder_type 合法
        assert encoder_type in ['scratch', 'spawnnet']
                
        if self.encoder_type == 'scratch':
            # 在 scratch 模式下，最终得到的特征维度为 32 通道、空间尺寸约为 35x35
            # 这里直接把展平后的维度保存下来，供后续网络使用
            self.repr_dim = 32 * 35 * 35
            # 简单的 4 层卷积网络：
            # - 第 1 层 stride=2，用于下采样
            # - 其余层 stride=1，保持空间尺寸，逐步提取特征
            self.convnet = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 3, stride=2),  # 输入通道为 obs_shape[0]，输出 32 通道
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1),
                nn.ReLU()
            )
            # 使用统一的初始化策略初始化所有参数
            self.apply(utils.weight_init)
        
        if self.encoder_type == 'spawnnet':
            # 预训练的 ResNet18 作为“教师”特征提取器，只用其前几层特征
            self.pretrained_resnet = models.resnet18(pretrained=True)
            if resnet_fix:
                # 如果选择固定预训练网络，则不更新其参数
                for params in self.pretrained_resnet.parameters():
                    params.requires_grad = False
            # 预训练分支大致输出形状: 64 * 21 * 21 -> 128 * 11 * 11（经过后续卷积）
            
            # scratch 分支的第一段卷积：从原始图像中提取低层视觉特征
            self.scratch_convnet_layer1 = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 84 * 84 -> 42 * 42
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 42 * 42 -> 21 * 21
            )
            # scratch 分支的第二段卷积：在与预训练特征融合后，进一步提取高层特征
            self.scratch_convnet_layer2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),  # 21 * 21 -> 21 * 21
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 21 * 21 -> 11 * 11
            )
            # scratch 分支形状从 32 * 21 * 21 -> 64 * 11 * 11
            # 下面是一系列 1x1 卷积和残差块，用于与预训练特征对齐和融合
            self.oneXone_conv_layer1 = nn.Sequential(
                nn.Conv2d(64, 32, 1, stride=1)  # 将预训练特征通道压缩到 32
            )
            self.oneXone_conv_layer2 = nn.Sequential(
                nn.Conv2d(128, 64, 1, stride=1)  # 再次调整通道数，便于与 scratch 分支融合
            )
            # 第一个残差块：在 64 通道上堆叠两个 3x3 卷积
            self.residual_conv_layer1 = nn.Sequential(
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                nn.ReLU()
            )
            # 第二个残差块：在 128 通道上堆叠两个 3x3 卷积
            self.residual_conv_layer2 = nn.Sequential(
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.ReLU()
            )
            
            # feature_dim 代表展开前的总特征维度：
            # - obs_shape[0] // 9 约等于相机数量（每个相机有 3 * time_steps 通道，这里 time_steps=3）
            # - 乘以 4 是因为在时间上拼接了 current / previous 差分等 4 份特征
            # - 128 * 11 * 11 是每一份特征的空间维度和通道数
            feature_dim = obs_shape[0]//9 * 4 * 128 * 11 * 11
            # 编码器输出的最终维度固定为 1024，方便下游网络设计
            self.repr_dim = 1024
            # 输出层：先做 LayerNorm 再做线性变换，将高维特征压缩到 repr_dim
            self.output_layer = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, self.repr_dim)
            )
            # 同样使用统一的初始化方法
            self.apply(utils.weight_init)

    def forward(self, x):
        """
        前向传播，将输入图像批次编码为一维特征向量。

        参数:
            x (torch.Tensor): 输入张量，形状为 [B, C, H, W]，
                              其中 C 与初始化时的 obs_shape[0] 一致。

        返回:
            torch.Tensor: 编码后的特征，形状为 [B, repr_dim]。
        """
        # 将像素从 [0, 255] 归一化到约 [-0.5, 0.5]，便于网络训练稳定
        x = x / 255.0 - 0.5
        
        if self.encoder_type == 'scratch':
            # 直接用自建卷积网络抽取特征，输出形状约为 [B, 32, 35, 35]
            h = self.convnet(x)  # shape: [B, D, H', W']
            # 展平成一维向量 [B, 32 * 35 * 35]，与 self.repr_dim 对齐
            h = h.view(h.shape[0], -1)
            return h
        
        if self.encoder_type == 'spawnnet':
            # batch size
            bsz = x.shape[0]
            # 通道数除以 9（3 通道 * 3 时间步）得到相机数量
            n_camera = x.shape[1] // 9
            # 先将通道维还原为 [B * n_camera * time_steps, 3, H, W] 的单图像形式
            x = x.view(-1, 3, 3, x.shape[2], x.shape[3]).view(-1, 3, x.shape[2], x.shape[3])
            
            # 预训练分支使用 detach 以避免梯度回流（当 resnet_fix=True 时尤其有效）
            hidden_pretrained = x.detach()
            # scratch 分支保留梯度
            hidden_scratch = x
            
            # Layer 1: 预训练 ResNet 仅前向到 layer1
            with torch.no_grad():
                for name, module in self.pretrained_resnet._modules.items():
                    hidden_pretrained = module(hidden_pretrained)
                    if name == "layer1":  # 到 layer1 停止
                        break
            # scratch 分支通过自建的第一段卷积
            hidden_scratch = self.scratch_convnet_layer1(hidden_scratch)
            
            # 使用 1x1 卷积对预训练特征做通道变换，并加 ReLU 非线性
            X_pretrained = torch.nn.functional.relu(self.oneXone_conv_layer1(hidden_pretrained))
            # 按通道拼接预训练特征与 scratch 特征，预训练特征乘以 pretrained_factor 控制其权重
            X_scratch = torch.cat([X_pretrained * self.pretrained_factor, hidden_scratch], dim=1)
            # 通过第一个残差块做特征变换，并加上残差提升表达能力
            X_scratch = X_scratch + self.residual_conv_layer1(X_scratch)
            hidden_scratch = X_scratch
            
            # Layer 2: 继续让预训练分支前向到 layer2
            with torch.no_grad():
                flag = False
                for name, module in self.pretrained_resnet._modules.items():
                    if flag:
                        hidden_pretrained = module(hidden_pretrained)
                        if name == "layer2":  # 到 layer2 停止
                            break
                    else:
                        if name == "layer1":
                            flag = True
            # scratch 分支通过第二段卷积，进一步下采样和提取高层特征
            hidden_scratch = self.scratch_convnet_layer2(hidden_scratch)
            
            # 再次用 1x1 卷积调整预训练特征通道数
            X_pretrained = torch.nn.functional.relu(self.oneXone_conv_layer2(hidden_pretrained))
            # 预训练特征与 scratch 特征再次融合
            X_scratch = torch.cat([X_pretrained * self.pretrained_factor, hidden_scratch], dim=1)
            # 第二个残差块，输出通道约为 128
            X_scratch = X_scratch + self.residual_conv_layer2(X_scratch)
            hidden_scratch = X_scratch
            
            # 此时 hidden_scratch 形状为: [n_camera * B * time_steps, 128, 11, 11]
            time_steps = 3
            # 重新组织维度: [n_camera * B, time_steps, C, H, W]
            X = hidden_scratch.view(-1, time_steps, hidden_scratch.shape[1],
                                    hidden_scratch.shape[2], hidden_scratch.shape[3])
            # 取当前时刻（去掉最早一个时间步），形状 [n_camera * B, time_steps-1, C, H, W]
            X_current = X[:, 1:, ...]
            # 使用差分构造“运动特征”：当前帧减去前一帧（前一帧梯度截断）
            X_previous = X_current - X[:, :time_steps - 1, ...].detach()
            # 将当前特征与差分特征在时间维拼接，得到 2 * (time_steps-1) 个时间通道
            X = torch.cat([X_current, X_previous], dim=1)
            # 将相机与时间维展开回 batch 维度，得到 [B, 4 * n_camera * 128, 11, 11]
            X = X.view(bsz, -1, X.shape[3], X.shape[4])
            # 再展平成一维向量 [B, feature_dim]
            X = X.view(bsz, -1)
            
            # 通过输出层 (LayerNorm + Linear) 映射到固定维度 repr_dim
            return self.output_layer(X)
            

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, moe_gate_dim=256, moe_hidden_dim=256, num_experts=32, top_k=4, dropout=0.1):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy1 = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                     nn.ReLU(inplace=True))

        self.policy2 = nn.Sequential(nn.ReLU(inplace=True),
                                     nn.Linear(hidden_dim, action_shape[0]))
          
        self.moe = MoE( input_dim=hidden_dim,
                        output_dim=hidden_dim,
                        gate_dim=moe_gate_dim,
                        hidden_dim=moe_hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        dropout=dropout,
                    )     
            
        self.apply(utils.weight_init)

    def forward(self, obs, std, metrics=None):
        h = self.trunk(obs)
        x = self.policy1(h)
        x, aux_loss = self.moe(x, metrics)
        
        mu = self.policy2(x)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist, aux_loss


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        obs = obs.reshape(obs.shape[0], -1)
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class VNetwork(nn.Module):
    def __init__(self, repr_dim, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.V = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs.reshape(obs.shape[0], -1)
        h = self.trunk(obs)
        v = self.V(h)
        return v


class MENTORAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, dormant_threshold,
                 target_dormant_ratio, dormant_temp, target_lambda,
                 lambda_temp, perturb_interval, min_perturb_factor,
                 max_perturb_factor, perturb_rate, num_expl_steps, stddev_type,
                 stddev_schedule, stddev_clip, expectile, use_tb,
                 lr_actor_ratio, aux_loss_scale_warmup, aux_loss_scale_warmsteps,
                 aux_loss_scale, aux_loss_type, encoder_type, resnet_fix,
                 oneXone_reg_scale, oneXone_reg_ratio, pretrained_factor, tp_set_size,
                 moe_gate_dim, moe_hidden_dim, num_experts, top_k, dropout):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_type = stddev_type
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.dormant_threshold = dormant_threshold
        self.target_dormant_ratio = target_dormant_ratio
        self.dormant_temp = dormant_temp
        self.target_lambda = target_lambda
        self.lambda_temp = lambda_temp
        self.dormant_ratio = 1
        self.perturb_interval = perturb_interval
        self.min_perturb_factor = min_perturb_factor
        self.max_perturb_factor = max_perturb_factor
        self.perturb_rate = perturb_rate
        self.expectile = expectile
        self.awaken_step = None
        self.aux_loss_scale_warmup = aux_loss_scale_warmup
        self.aux_loss_scale_warmsteps = aux_loss_scale_warmsteps
        self.aux_loss_scale_max = aux_loss_scale
        self.aux_loss_scale = self.calc_aux_loss_scale(0)
        self.lr_actor_ratio = lr_actor_ratio
        self.aux_loss_type = aux_loss_type
        self.oneXone_reg_scale = oneXone_reg_scale
        self.oneXone_reg_ratio = oneXone_reg_ratio
        self.pretrained_factor = pretrained_factor
        self.pretrained_factor = pretrained_factor
        self.tp_set_size = tp_set_size
        self.moe_gate_dim = moe_gate_dim
        self.moe_hidden_dim = moe_hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout

        # models
        self.encoder = Encoder(obs_shape, encoder_type, resnet_fix, pretrained_factor).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, 
                           moe_gate_dim, moe_hidden_dim, num_experts, top_k, dropout).to(device)
        self.value_predictor = VNetwork(self.encoder.repr_dim, feature_dim, hidden_dim).to(device)
        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr * (1. if encoder_type == 'scratch' else 0.5))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr * self.lr_actor_ratio)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.predictor_opt = torch.optim.Adam(self.value_predictor.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.n_updates = 0
        self.perturb_time = 0
        self.train()
        self.critic_target.train()
        
        # Task-oriented Perturbation
        self.tp_set = utils.models_tuple(maxsize=self.tp_set_size, moe=True, gate=True)

    @property
    def dormant_stddev(self):
        return 0.8 / (1 + math.exp(-self.dormant_temp * (self.dormant_ratio - self.target_dormant_ratio)))

    def stddev(self, step):
        return self.dormant_stddev

    def perturb_factor(self):
        return min(max(self.min_perturb_factor, 1 - self.perturb_rate * self.dormant_ratio), self.max_perturb_factor)

    @property
    def lambda_(self):
        return self.target_lambda / (1 + math.exp(self.lambda_temp * (self.dormant_ratio - self.target_dormant_ratio)))

    def calc_aux_loss_scale(self, step):
        if self.aux_loss_scale_warmup < 0 or self.aux_loss_scale_warmsteps < 0:
            return self.aux_loss_scale_max
        if step > self.aux_loss_scale_warmsteps:
            return self.aux_loss_scale_max
        return math.exp(
            math.log(self.aux_loss_scale_warmup) +  step / self.aux_loss_scale_warmsteps * ( math.log(self.aux_loss_scale_max) - math.log(self.aux_loss_scale_warmup) )
        )

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.value_predictor.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        dist, _ = self.actor(obs, self.stddev(step))
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_predictor(self, obs, action):
        metrics = dict()

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        V = self.value_predictor(obs)
        vf_err = V - Q
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (1 - self.expectile)
        predictor_loss = (vf_weight * (vf_err**2)).mean()

        if self.use_tb:
            metrics['predictor_loss'] = predictor_loss.item()

        self.predictor_opt.zero_grad(set_to_none=True)
        predictor_loss.backward()
        self.predictor_opt.step()

        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            dist, _ = self.actor(next_obs, self.stddev(step))
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V_explore = torch.min(target_Q1, target_Q2)
            target_V_exploit = self.value_predictor(next_obs)
            target_V = self.lambda_ * target_V_exploit + (1 - self.lambda_) * target_V_explore
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        
        if self.oneXone_reg_scale > 0.01:
            def customized_regularization(weight):
                regularization = torch.norm(weight, p=2)
                return regularization
            critic_loss += self.oneXone_reg_scale * customized_regularization(self.encoder.oneXone_conv_layer1.weight, self.oneXone_reg_ratio)
            critic_loss += self.oneXone_reg_scale * customized_regularization(self.encoder.oneXone_conv_layer2.weight, self.oneXone_reg_ratio)
            
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()
        dist, aux_loss = self.actor(obs, self.stddev(step), metrics)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        (actor_loss + aux_loss).backward()
        grad_mean = torch.mean(torch.abs(torch.cat([p.grad.flatten() for p in self.actor.parameters() if p.grad is not None])))
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.001)
        self.actor_opt.step()

        if self.use_tb:
            metrics['aux_loss'] = aux_loss.item()
            metrics['actor_grad_mean'] = grad_mean.item()
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def perturb(self):
        utils.perturb(self.actor, self.actor_opt, self.perturb_factor(), tp_set=self.tp_set, name="actor")
        # utils.perturb(self.actor.moe.experts, self.actor_opt, self.perturb_factor(), tp_set=self.tp_set, name="actor_moe_expert")
        # utils.perturb(self.actor.moe.gate, self.actor_opt, self.perturb_factor(), tp_set=self.tp_set, name="actor_moe_gate")
        utils.perturb(self.critic, self.critic_opt, self.perturb_factor(), tp_set=self.tp_set, name="critic")
        utils.perturb(self.critic_target, self.critic_opt, self.perturb_factor(), tp_set=self.tp_set, name="critic_target")
        utils.perturb(self.value_predictor, self.predictor_opt, self.perturb_factor(), tp_set=self.tp_set, name="value_predictor")
        # utils.perturb(self.encoder, self.encoder_opt, self.perturb_factor())

    def update(self, replay_iter, step):
        metrics = dict()

        self.n_updates += 1

        # aux_loss_scale
        self.aux_loss_scale = self.calc_aux_loss_scale(self.n_updates)

        # perturb
        if self.perturb_interval > 0 and self.n_updates % self.perturb_interval == 0:
            self.perturb()
            self.perturb_time += 1

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # calculate dormant ratio
        self.dormant_ratio, metrics = utils.cal_dormant_ratio(self.actor, obs.detach(), 0,\
            percentage=self.dormant_threshold, metrics=metrics)

        if self.awaken_step is None and step > self.num_expl_steps and self.dormant_ratio < self.target_dormant_ratio:
            self.awaken_step = step

        if self.use_tb:
            metrics['perturb_time'] = self.perturb_time
            metrics['batch_reward'] = reward.mean().item()
            metrics['actor_dormant_ratio'] = self.dormant_ratio
            metrics['aux_loss_scale'] = self.aux_loss_scale
        
        # update predictor
        metrics.update(self.update_predictor(obs.detach(), action))

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics
