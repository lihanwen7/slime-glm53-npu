"""Memory-efficient NPU ClampedSwiGLU; FP32 arithmetic, input-dtype storage."""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _forward(X, Y, N: tl.constexpr, H: tl.constexpr, LIMIT: tl.constexpr, B: tl.constexpr):
    for start in range(tl.program_id(0) * B, N, tl.num_programs(0) * B):
        i = start + tl.arange(0, B)
        offset = (i // H) * (2 * H) + i % H
        gate = tl.load(X + offset, i < N, other=0).to(tl.float32)
        up = tl.load(X + offset + H, i < N, other=0).to(tl.float32)
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        tl.store(Y + i, (gate * sigmoid) * up, i < N)


@triton.jit
def _backward(X, DY, DX, N: tl.constexpr, H: tl.constexpr, LIMIT: tl.constexpr, B: tl.constexpr):
    for start in range(tl.program_id(0) * B, N, tl.num_programs(0) * B):
        i = start + tl.arange(0, B)
        offset = (i // H) * (2 * H) + i % H
        raw_gate = tl.load(X + offset, i < N, other=0).to(tl.float32)
        raw_up = tl.load(X + offset + H, i < N, other=0).to(tl.float32)
        dy = tl.load(DY + i, i < N, other=0).to(tl.float32)
        gate = tl.minimum(raw_gate, LIMIT)
        up = tl.minimum(tl.maximum(raw_up, -LIMIT), LIMIT)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        dg = (dy * up) * (sigmoid * (1.0 + gate * (1.0 - sigmoid)))
        du = dy * (gate * sigmoid)
        # torch.clamp passes the gradient at both exact boundaries.
        dg = tl.where(raw_gate <= LIMIT, dg, 0.0)
        du = tl.where((raw_up >= -LIMIT) & (raw_up <= LIMIT), du, 0.0)
        tl.store(DX + offset, dg, i < N)
        tl.store(DX + offset + H, du, i < N)


@lru_cache(maxsize=None)
def _cores(device_index):
    from triton.runtime import driver

    return driver.active.utils.get_device_properties(device_index)["num_vectorcore"]


class _ClampedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, limit):
        h = x.shape[-1] // 2
        y = torch.empty((*x.shape[:-1], h), device=x.device, dtype=x.dtype)
        ctx.save_for_backward(x)
        ctx.limit = limit
        if y.numel():
            with torch.npu.device(x.device):
                grid = (min(triton.cdiv(y.numel(), 2048), _cores(x.device.index)),)
                _forward[grid](x, y, y.numel(), h, limit, 2048, enable_fp_fusion=False)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        if grad_output.numel():
            dy = grad_output.contiguous()
            with torch.npu.device(x.device):
                grid = (min(triton.cdiv(dy.numel(), 2048), _cores(x.device.index)),)
                _backward[grid](x, dy, dx, dy.numel(), x.shape[-1] // 2, ctx.limit, 2048,
                                enable_fp_fusion=False)
        return dx, None


def clamped_swiglu(x, limit):
    """Fused first-order training path for contiguous FP16/BF16/FP32 NPU inputs."""
    return _ClampedSwiGLU.apply(x, limit)
