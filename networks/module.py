import torch
import torch.nn as nn
import torch.nn.functional as F
import sys  # NOQA
import math  # NOQA
import numpy as np
import random
from functools import partial  # NOQA
from typing import Optional, Callable, Any  # NOQA
from einops import rearrange  # NOQA

import torch.utils.checkpoint as checkpoint  # NOQA
from timm.models.layers import DropPath, trunc_normal_  # NOQA
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count  # NOQA

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:  # NOQA
    from csm_triton import cross_scan_fn, cross_merge_fn

try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:  # NOQA
    from csms6s import selective_scan_fn, selective_scan_flop_jit  # NOQA

# FLOPs counter not prepared fro mamba2
try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except:  # NOQA
    from mamba2.ssd_minimal import selective_scan_chunk_fn  # NOQA

sys.path.append("..")


# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3,  # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        forward_type="v05_noz",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear

        # tags for forward_type ==============================
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_core, force_fp32=(not self.disable_force32), selective_scan_backend="mamba", scan_force_torch=True),
            v02=partial(self.forward_core, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            v03=partial(self.forward_core, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            v04=partial(self.forward_core, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"
            v05=partial(self.forward_core, force_fp32=False, no_einsum=True),  # selective_scan_backend="oflex", scan_mode="cross2d"
            # ===============================
            v051d=partial(self.forward_core, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            v052d=partial(self.forward_core, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            v052dc=partial(self.forward_core, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            # ===============================
            v2=partial(self.forward_core, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_core, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
        )

    def forward_core(
        self,
        x: torch.Tensor = None,
        # ==============================
        force_fp32=False,  # True: input fp32
        # ==============================
        ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend=None,
        # ==============================
        # ==============================
        scan_mode="cross2d",
        scan_force_torch=False,
        # ==============================
        **kwargs,
    ):
        assert scan_mode in ["unidi", "bidi", "cross2d", "cascade2d"]
        assert selective_scan_backend in [None, "oflex", "core", "mamba", "torch"]
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)  # NOQA

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)

        if _scan_mode == 3:
            x_proj_bias = getattr(self, "x_proj_bias", None)

            def scan_rowcol(
                x: torch.Tensor,
                proj_weight: torch.Tensor,
                proj_bias: torch.Tensor,
                dt_weight: torch.Tensor,
                dt_bias: torch.Tensor,  # (2*c)
                _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                _Ds: torch.Tensor,
                width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1), bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1, 2).contiguous()  # added0510 to avoid nan
            y_row = scan_rowcol(
                x,
                proj_weight=self.x_proj_weight.view(4, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias=(self.dt_projs_bias.view(4, -1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)  # (B,C,H,W)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1, 2).contiguous()  # added0510 to avoid nan
            y_col = scan_rowcol(
                y_row,
                proj_weight=self.x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(self.dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if self.dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)

            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        forward_type="v05_noz",
        # =============================
        use_checkpoint: bool = False,
        # =============================
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            # basic dims ===========
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # dwconv ===============
            d_conv=ssm_conv,  # < 2 means no conv
            conv_bias=ssm_conv_bias,
            # ======================
            dropout=ssm_drop_rate,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            forward_type=forward_type,
            channel_first=channel_first,
            # ======================
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input
        if self.use_checkpoint:
            return checkpoint.checkpoint(self.forward, x)
        else:
            x = x + self.drop_path(self.self_attention(self.norm(x)))
        return x


class VSS(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim=96,
        depth=2,
        drop_path=[0.1, 0.1],
        use_checkpoint=False,
        norm_layer=nn.LayerNorm,
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        forward_type="v2",
        # ===========================
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                forward_type=forward_type,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class DynamicReLU(nn.Module):
    def forward(self, x):
        alpha = torch.sigmoid(x.mean(dim=(2, 3), keepdim=True))  # 动态斜率
        return torch.max(x, alpha * x)  # 改进版LeakyReLU


class MambaConv2d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(
        self,
        in_channels=8,
        square_kernel_size=3,
        band_kernel_size=11,
        vss_depth=1,
        groups=4,
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        forward_type="v05_noz",
        drop_path_rate=0.1,
        norm_layer="LN2D",  # "BN", "LN2D"
        use_checkpoint=False,
        # =========================
        **kwargs,
    ):
        super(MambaConv2d, self).__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate * 0.02, vss_depth)]
        gc = int(in_channels // groups)  # channel numbers of a convolution branch
        self.groups = groups
        self.spatial_conv = VSS(
            dim=gc,
            depth=vss_depth,
            drop_path=dpr,
            use_checkpoint=use_checkpoint,
            norm_layer=norm_layer,
            channel_first=self.channel_first,
            # =================
            ssm_d_state=math.ceil(gc / 6) if ssm_d_state is None else ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            forward_type=forward_type,
        )
        self.dwconv_hw = Conv2d(gc, gc, kernel_size=square_kernel_size, stride=1, padding=square_kernel_size // 2)
        self.dwconv_w = Conv2d(gc, gc, kernel_size=(1, band_kernel_size), stride=(1, 1), padding=(0, band_kernel_size // 2))
        self.dwconv_h = Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), stride=(1, 1), padding=(band_kernel_size // 2, 0))
        self.split_indexes = (gc, gc, gc, gc)
        self.conv2d = Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = x
        x_hw, x_w, x_h, x_s = torch.split(x, self.split_indexes, dim=1)
        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)
        x_s = self.spatial_conv(x_s)
        x = y + torch.cat([x_hw, x_w, x_h, x_s], dim=1)
        x = self.conv2d(x)
        return x


# class InceptionConv3d(nn.Module):
#     """ Inception depthweise convolution
#     """
#     def __init__(
#         self,
#         in_channels=8,
#         square_kernel_size=3,
#         band_kernel_size=11,
#         vss_depth=1,
#         groups=4,
#         # =========================
#         ssm_d_state=16,
#         ssm_ratio=2.0,
#         ssm_dt_rank="auto",
#         ssm_act_layer="silu",
#         ssm_conv=3,
#         ssm_conv_bias=True,
#         ssm_drop_rate=0.0,
#         forward_type="v05_noz",
#         drop_path_rate=0.1,
#         norm_layer="LN2D",  # "BN", "LN2D"
#         use_checkpoint=False,
#         # =========================
#         **kwargs,
#     ):
#         super(InceptionConv3d, self).__init__()
#         self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])

#         _NORMLAYERS = dict(
#             ln=nn.LayerNorm,
#             ln2d=LayerNorm2d,
#             bn=nn.BatchNorm2d,
#         )
#         _ACTLAYERS = dict(
#             silu=nn.SiLU,
#             gelu=nn.GELU,
#             relu=nn.ReLU,
#             sigmoid=nn.Sigmoid,
#         )
#         norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
#         ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate * 0.02, vss_depth)]
#         gc = int(in_channels // groups)  # channel numbers of a convolution branch
#         self.groups = groups
#         self.spatial_conv = VSS(
#             dim=gc,
#             depth=vss_depth,
#             drop_path=dpr,
#             use_checkpoint=use_checkpoint,
#             norm_layer=norm_layer,
#             channel_first=self.channel_first,
#             # =================
#             ssm_d_state=math.ceil(gc / 6) if ssm_d_state is None else ssm_d_state,
#             ssm_ratio=ssm_ratio,
#             ssm_dt_rank=ssm_dt_rank,
#             ssm_act_layer=ssm_act_layer,
#             ssm_conv=ssm_conv,
#             ssm_conv_bias=ssm_conv_bias,
#             ssm_drop_rate=ssm_drop_rate,
#             forward_type=forward_type,
#         )
#         self.dwconv_d = nn.Conv3d(gc, gc, kernel_size=(1, square_kernel_size, square_kernel_size), padding=(0, square_kernel_size // 2, square_kernel_size // 2))
#         self.dwconv_w = nn.Conv3d(gc, gc, kernel_size=(1, 1, band_kernel_size), padding=(0, 0, band_kernel_size // 2))
#         self.dwconv_h = nn.Conv3d(gc, gc, kernel_size=(1, band_kernel_size, 1), padding=(0, band_kernel_size // 2, 0))
#         self.split_indexes = (gc, gc, gc, gc)
#         self.conv3d = Conv3d(in_channels, in_channels, kernel_size=square_kernel_size, stride=1, padding=1)

#     def forward(self, x):
#         y = x
#         x_d, x_w, x_h, x_s = torch.split(x, self.split_indexes, dim=1)
#         x_d = self.dwconv_d(x_d)
#         x_w = self.dwconv_w(x_w)
#         x_h = self.dwconv_h(x_h)
#         B, C, D, H, W = x_s.shape
#         x_s = x_s.permute(0, 2, 1, 3, 4).reshape(B*D, C, H, W)
#         x_s = self.spatial_conv(x_s)
#         x_s = x_s.reshape(B, D, C, H, W).permute(0, 2, 1, 3, 4)
#         x = y + torch.cat([x_d, x_w, x_h, x_s], dim=1)  # 原始特征，提取后的高层次特征会不会进行互补呢？
#         x = self.conv3d(x)
#         return x


class InceptionConv3d(nn.Module):
    """ Inception depthweise convolution, 为了节约显存，这里仅仅使用nn.Conv3d
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, groups=4):
        super().__init__()
        self.groups = groups
        gc = int(in_channels // groups)  # channel numbers of a convolution branch
        self.dwconv_d = nn.Conv3d(gc, gc, kernel_size=(1, square_kernel_size, square_kernel_size), padding=(0, square_kernel_size // 2, square_kernel_size // 2))
        self.dwconv_d1 = nn.Conv3d(gc, gc, kernel_size=(1, 7, 7), padding=(0, 7 // 2, 7 // 2))
        self.dwconv_w = nn.Conv3d(gc, gc, kernel_size=(1, 1, band_kernel_size), padding=(0, 0, band_kernel_size // 2))
        self.dwconv_h = nn.Conv3d(gc, gc, kernel_size=(1, band_kernel_size, 1), padding=(0, band_kernel_size // 2, 0))
        self.split_indexes = (gc, gc, gc, gc)
        self.conv3d = Conv3d(in_channels, in_channels, kernel_size=square_kernel_size, stride=1, padding=1)

    def forward(self, x):
        y = x
        # x_d0, x_w, x_h, x_d = torch.split(x, self.split_indexes, dim=1)
        x_d0, x_d, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x_d0 = self.dwconv_d(x_d0)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)
        x_d = self.dwconv_d1(x_d)
        x = y + torch.cat([x_d0, x_d, x_w, x_h], dim=1)  # 原始特征，提取后的高层次特征会不会进行互补呢？
        x = self.conv3d(x)
        return x


def channel_shuffle(x, groups):
    """
    支持3D (B, C, D, H, W) 和 4D (B, C, T, H, W) 输入的通道混洗
    Args:
        x: 输入张量，形状为 [B, C, (T/D), H, W]
        groups: 分组数（必须能被通道数整除）
    Returns:
        混洗后的张量（保持输入维度）
    """
    batch_size, num_channels, *spatial_dims = x.size()
    channels_per_group = num_channels // groups

    # 检查输入维度合法性
    if len(spatial_dims) not in [2, 3]:
        raise ValueError(f"输入必须是3D或4D张量，但得到形状: {x.shape}")

    # 统一处理：将空间维度合并为一个维度
    x = x.view(batch_size, groups, channels_per_group, *spatial_dims)

    # 交换组和子通道维度
    x = x.permute(0, 2, 1, *range(3, 3 + len(spatial_dims))).contiguous()

    # 恢复原始形状
    return x.view(batch_size, num_channels, *spatial_dims)


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)], indexing='ij')  # 这里生成的是索引，所以先h，再w
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xyz[:, 2:3][proj_xyz[:, 2:3] == 0] += 0.00001  # NAN BUG, not on dtu, but on blendedmvs

        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=False).type(torch.float32)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


def winner_take_all(prob_volume, depth_values):
    """
    :param prob_volume: (b, d, h, w)
    :param depth_values: (b, d, h, w)
    :return: (b, h, w)
    """
    _, idx = torch.max(prob_volume, dim=1, keepdim=True)
    depth = torch.gather(depth_values, 1, idx).squeeze(1)
    return depth


def unity_regression(prob_volume, depth_values, interval):
    """
    :param interval: (b, )
    :param prob_volume: (b, d, h, w)
    :param depth_values: (b, d, h, w)
    :return: (b, h, w)
    """
    val, idx = torch.max(prob_volume, dim=1, keepdim=True)

    wta_depth = torch.gather(depth_values, 1, idx)
    offset = (1 - val) * interval

    depth = wta_depth + offset
    depth = depth.squeeze(1)

    return depth


def get_cur_depth_range_samples(last_depth, ndepth, depth_inteval_pixel):
    # cur_depth: (B, H, W)
    # return depth_range_values: (B, D, H, W)
    last_depth_min = (last_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    last_depth_max = (last_depth + ndepth / 2 * depth_inteval_pixel)
    new_interval = (last_depth_max - last_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = last_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=last_depth.device,
                                                                      dtype=last_depth.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1,
                                                                                                   1) * new_interval.unsqueeze(1))

    return depth_range_samples, (ndepth * depth_inteval_pixel) / (ndepth - 1)


def get_depth_range_samples(last_depth, ndepth, depth_inteval_pixel, shape=None):
    # cur_depth: (B, H, W) or (B, D)
    # return depth_range_samples: (B, D, H, W)

    if last_depth.dim() == 2:
        last_depth_min = last_depth[:, 0]  # (B,)
        last_depth_max = last_depth[:, -1]
        new_interval = (last_depth_max - last_depth_min) / (ndepth - 1)  # (B, )
        stage_interval = new_interval[0]

        depth_range_samples = last_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=last_depth.device, dtype=last_depth.dtype,
                                                                          requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(
            1))  # (B, D)

        # (B, D, H, W)
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[0], shape[1])

    else:

        depth_range_samples, stage_interval = get_cur_depth_range_samples(last_depth, ndepth, depth_inteval_pixel)

    return depth_range_samples, stage_interval


def random_image_mask(img, filter_size):
    '''

    :param img: [B x 3 x H x W]
    :param crop_size:
    :return:
    '''
    fh, fw = filter_size
    _, _, h, w = img.size()

    if fh == h and fw == w:
        return img, None

    x = np.random.randint(0, w - fw)
    y = np.random.randint(0, h - fh)
    filter_mask = torch.ones_like(img)         # B x 3 x H x W
    filter_mask[:, :, y:y + fh, x:x + fw] = 0.0    # B x 3 x H x W
    img = img * filter_mask                    # B x 3 x H x W
    return img, filter_mask


def random_patch_mask(images, patch_size=32, mask_ratio=1 / 3):
    """
    向量化实现：随机选择每张图的 1/3 patch 完全遮挡，其余保留，返回 masked image 和 mask_map。

    Args:
        images: Tensor [B, C, H, W]
        patch_size: Patch 尺寸
        mask_ratio: 遮挡比例，例如 1/3

    Returns:
        masked_images: Tensor [B, C, H, W]
        mask_map: Tensor [B, 1, H, W]，值为 1 表示保留，0 表示遮挡
    """
    B, C, H, W = images.shape
    patch_size = random.choice(patch_size)
    assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size"

    nh, nw = H // patch_size, W // patch_size
    N = nh * nw  # 每张图像的 patch 数
    num_mask = int(mask_ratio * N)

    device = images.device

    # unfold 为 patch：[B, N, C, ps, ps]
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, N, C, patch_size, patch_size)

    # 向量化创建随机排序索引：[B, N]
    rand_scores = torch.rand(B, N, device=device)
    sorted_idx = rand_scores.argsort(dim=1)

    # 构造 mask_flat：对每张图前 num_mask 个 patch 设置为 0，其余为 1
    mask_flat = torch.ones(B, N, device=device)
    row_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, num_mask)  # [B, num_mask]
    col_idx = sorted_idx[:, :num_mask]  # [B, num_mask]
    mask_flat[row_idx, col_idx] = 0  # 设置遮挡 patch 为 0

    # 应用到 patch 上
    mask_weight = mask_flat.view(B, N, 1, 1, 1)  # [B, N, 1, 1, 1]
    masked_patches = patches * mask_weight

    # reshape 回原图
    masked_patches = masked_patches.view(B, nh, nw, C, patch_size, patch_size)
    masked_patches = masked_patches.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

    # 构造 mask_map: [B, 1, H, W]
    mask_map = mask_flat.view(B, nh, nw, 1, 1).expand(-1, -1, -1, patch_size, patch_size)
    mask_map = mask_map.permute(0, 3, 1, 4, 2).reshape(B, 1, H, W)

    return masked_patches, mask_map


def depth_regression(p, depth_hypotheses):
    if depth_hypotheses.dim() <= 2:
        # print("regression dim <= 2")
        depth_hypotheses = depth_hypotheses.view(*depth_hypotheses.shape, 1, 1)
    depth = torch.sum(p * depth_hypotheses, 1)

    return depth


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return
