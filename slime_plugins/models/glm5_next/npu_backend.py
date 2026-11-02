"""Functional Ascend backend for the native GLM5-next model.

KDA and causal convolution use AscendC forward/backward bridges. Gated
normalization uses PyTorch NPU operations; sparse MLA uses CANN AscendC.
CP gathers complete packed streams before recurrent operations: this validates
functionality, not FLA's GPU CP performance or fused-kernel performance.
"""

import os

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region


def _boundaries(cu, length):
    boundaries = [0, length] if cu is None else cu.tolist()
    if boundaries[0] != 0 or boundaries[-1] != length:
        raise ValueError(f"Packed boundaries do not cover {length} tokens: {boundaries}")
    return boundaries


def _gather(x, context):
    if context is None:
        return x
    return gather_from_sequence_parallel_region(x.transpose(0, 1).contiguous(), group=context.group).transpose(0, 1)


def _local(x, context):
    if context is None:
        return x
    return x.chunk(context.group.size(), dim=1)[context.group.rank()].contiguous()


def build_cp_context(cu_seqlens, group, conv1d_kernel_size):
    return SimpleNamespace(cu_seqlens=cu_seqlens, group=group)


def _convolve(x, weight, activation, cu):
    # FLA 26.10 provides Conv1d directly; do not gate it on the legacy vendor.
    # Keep an explicit torch override for numerical comparisons.
    backend = os.environ.get("GLM53_CONV_BACKEND", "ascendc")
    if backend not in ("ascendc", "torch"):
        raise ValueError(f"Unsupported GLM53_CONV_BACKEND: {backend}")
    if backend == "ascendc" and x.device.type == "npu" and x.dtype in (torch.bfloat16, torch.float16):
        from .ascendc_conv1d import causal_conv1d_ascendc

        return causal_conv1d_ascendc(x, weight, activation, cu)
    return _convolve_torch(x, weight, activation, cu)


def _convolve_torch(x, weight, activation, cu):
    outputs = []
    for start, end in zip((bounds := _boundaries(cu, x.shape[1]))[:-1], bounds[1:], strict=True):
        if start == end:
            continue
        part = x[:, start:end].transpose(1, 2)
        with torch.profiler.record_function("glm53::conv_fallback"):
            part = F.pad(part, (weight.shape[-1] - 1, 0))
            part = F.conv1d(part, weight[:, None], groups=weight.shape[0]).transpose(1, 2)
        if activation == "silu":
            part = F.silu(part)
        elif activation is not None:
            raise ValueError(f"Unsupported convolution activation: {activation}")
        outputs.append(part)
    return torch.cat(outputs, dim=1)


class ShortConvolution(torch.nn.Module):
    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu"):
        super().__init__()
        if bias:
            raise ValueError("GLM5-next NPU convolution expects bias=False")
        self.kernel_size = (kernel_size,)
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, x, cu_seqlens=None):
        return _convolve(x, self.weight.squeeze(1), self.activation, cu_seqlens), None


def causal_conv1d(x, weight, activation, cp_context):
    out = _convolve(_gather(x, cp_context), weight, activation, cp_context.cu_seqlens)
    return _local(out, cp_context), None


class FusedRMSNormGated(torch.nn.Module):
    """Same parameter contract; use autograd-supported NPU operations."""

    def __init__(self, hidden_size, eps, activation="sigmoid"):
        super().__init__()
        if activation != "sigmoid":
            raise ValueError("GLM5-next gated norm expects sigmoid")
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, gate):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float() * gate.float().sigmoid()).to(x.dtype)


def chunk_kda(q, k, v, *, g, beta, cu_seqlens, cp_context=None, **kwargs):
    from .ascendc_kda import chunk_kda_ascendc

    q, k, v, g, beta = [_gather(x, cp_context).contiguous() for x in (q, k, v, g, beta)]
    cpu_bounds = kwargs.pop("cu_seqlens_cpu", None)
    bounds = list(cpu_bounds) if cpu_bounds is not None else _boundaries(cu_seqlens, q.shape[1])
    if not bounds or bounds[0] != 0 or bounds[-1] != q.shape[1] or any(b < a for a, b in zip(bounds, bounds[1:])):
        raise ValueError("KDA sequence boundaries must be ordered and cover the gathered stream.")
    mode = os.getenv("GLM53_KDA_SEQUENCE_MODE", "loop")
    if mode not in ("loop", "packed"):
        raise ValueError(f"Unsupported GLM53_KDA_SEQUENCE_MODE: {mode}")
    if mode == "packed":
        # Remove empty sequences without changing token positions.
        bounds = list(dict.fromkeys(bounds))
        with torch.profiler.record_function("glm53::kda_packed"):
            out, _ = chunk_kda_ascendc(q, k, v, g=g, beta=beta, cu_seqlens=bounds, **kwargs)
        return _local(out, cp_context), None
    outputs = []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        if start == end:
            continue
        out, _ = chunk_kda_ascendc(
            q[:, start:end], k[:, start:end], v[:, start:end],
            g=g[:, start:end], beta=beta[:, start:end], **kwargs,
        )
        outputs.append(out)
    return _local(torch.cat(outputs, dim=1), cp_context), None


class SparseMLA:
    @staticmethod
    def apply(q, kv, indices, scaling):
        # Keep CANN SMLA loading lazy for rollout/driver-only imports.
        from .ascendc_smla import SparseMLA as AscendCSparseMLA

        return AscendCSparseMLA.apply(q, kv, indices, scaling)
