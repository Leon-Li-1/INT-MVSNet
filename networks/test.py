class SRU2D(nn.Module):
    """
    空间重加权单元（SRU2D）
    对每个通道做可微的软门控，然后交叉相加。
    """
    def __init__(self,
                 out_channels: int,
                 num_groups: int = 4,
                 gate_threshold: float = 0.5,
                 eps: float = 1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups,
                               num_channels=out_channels,
                               eps=eps)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) GroupNorm
        gn_x = self.gn(x)

        # 2) 可微分的权重归一化
        w = self.gn.weight                               # [C]
        w_norm = w / (w.sum() + 1e-6)                    # [C]
        w_norm = w_norm.view(1, -1, 1, 1)                # 可广播到 [B,C,H,W]

        # 3) 软门控
        gates = self.sigmoid(gn_x * w_norm)              # in (0,1)

        # 4) 直通估计的硬阈值（可选）
        hard_mask = (gates > self.gate_threshold).float()
        # straight-through：正向用硬掩码，反向保留 gates 的梯度
        weight = 1 + 2 * (hard_mask + gates - gates.detach())
        x_mod = weight * x

        # 5) 交叉相加
        B, C, H, W = x_mod.shape
        assert C % 2 == 0, "通道数必须为偶数"
        x1, x2 = x_mod[:, :C//2], x_mod[:, C//2:]
        out = torch.cat([x1 + x2, x2 + x1], dim=1)
        return out


class CRU2D(nn.Module):
    """
    通道重标定单元（CRU2D）
    拆分通道后做分组卷积 + 点卷积，再用 sigmoid 做通道重标定。
    """
    def __init__(self,
                 out_channels: int,
                 alpha: float = 0.5,
                 squeeze_ratio: int = 2,
                 group_size: int = 2,
                 kernel_size: int = 3):
        super().__init__()
        self.up_ch = int(alpha * out_channels)
        self.low_ch = out_channels - self.up_ch

        # 通道压缩
        self.squeeze_up = nn.Conv2d(self.up_ch,
                                    self.up_ch // squeeze_ratio,
                                    kernel_size=1,
                                    bias=False)
        self.squeeze_low = nn.Conv2d(self.low_ch,
                                     self.low_ch // squeeze_ratio,
                                     kernel_size=1,
                                     bias=False)

        # 高频分支：分组卷积 + 点卷积
        self.group_conv = nn.Conv2d(self.up_ch // squeeze_ratio,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    padding=kernel_size//2,
                                    groups=group_size,
                                    bias=False)
        self.pw_up = nn.Conv2d(self.up_ch // squeeze_ratio,
                               out_channels,
                               kernel_size=1,
                               bias=False)

        # 低频分支：点卷积 + 残差
        self.pw_low = nn.Conv2d(self.low_ch // squeeze_ratio,
                                out_channels - self.low_ch // squeeze_ratio,
                                kernel_size=1,
                                bias=False)

        # 通道重标定
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 拆分通道
        up, low = torch.split(x, [self.up_ch, self.low_ch], dim=1)

        # 2) 通道压缩
        up_s = self.squeeze_up(up)
        low_s = self.squeeze_low(low)

        # 3) 分支变换
        y1 = self.group_conv(up_s) + self.pw_up(up_s)
        y2 = torch.cat([self.pw_low(low_s), low_s], dim=1)

        # 4) 拼接并做通道重标定
        fused = torch.cat([y1, y2], dim=1)               # [B, out_channels, H, W]
        weights = self.global_pool(fused)                 # [B, C, 1, 1]
        weights = self.channel_act(weights)               # sigmoid 标定
        out = fused * weights                             # 通道逐元素相乘

        # 5) 可选：一分为二再相加
        B, C, H, W = out.shape
        assert C % 2 == 0
        a, b = out[:, :C//2], out[:, C//2:]
        return a + b


class ScConv2D(nn.Module):
    """
    空间-通道卷积块：SRU2D → CRU2D 串联
    """
    def __init__(self,
                 out_channels: int,
                 # SRU 参数
                 num_groups: int = 4,
                 gate_threshold: float = 0.5,
                 # CRU 参数
                 alpha: float = 0.5,
                 squeeze_ratio: int = 2,
                 group_size: int = 2,
                 kernel_size: int = 3):
        super().__init__()
        self.sru = SRU2D(out_channels,
                         num_groups=num_groups,
                         gate_threshold=gate_threshold)
        self.cru = CRU2D(out_channels,
                         alpha=alpha,
                         squeeze_ratio=squeeze_ratio,
                         group_size=group_size,
                         kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sru(x)
        x = self.cru(x)
        return x