"""Ascend backend for the GLM5-next model.

KDA and causal convolution use AscendC forward/backward bridges. Gated
normalization uses PyTorch NPU operations; sparse MLA uses CANN AscendC.
CP exchanges sequence shards for head shards when the head count is divisible
by the CP size, and restores the local sequence layout after computation.
"""

import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel.mappings import all_to_all, gather_from_sequence_parallel_region


def _boundaries(cu, length):
    boundaries = [0, length] if cu is None else (cu.tolist() if hasattr(cu, "tolist") else list(cu))
    if length <= 0:
        raise ValueError("Packed input must contain at least one token")
    if len(boundaries) < 2 or any(not isinstance(value, int) for value in boundaries):
        raise ValueError("Packed boundaries must contain at least two integer offsets")
    if boundaries[0] != 0 or boundaries[-1] != length:
        raise ValueError(f"Packed boundaries do not cover {length} tokens: {boundaries}")
    if any(end < start for start, end in zip(boundaries, boundaries[1:], strict=False)):
        raise ValueError("Packed boundaries must be nondecreasing")
    return boundaries


def _gather(x, context):
    if context is None:
        return x
    return gather_from_sequence_parallel_region(x.transpose(0, 1).contiguous(), group=context.group).transpose(0, 1)


def _local(x, context):
    if context is None:
        return x
    return x.chunk(context.group.size(), dim=1)[context.group.rank()].contiguous()


def _can_shard_heads(x, context):
    return context is not None and x.shape[0] == 1 and x.shape[2] % context.group.size() == 0


def _to_head_parallel(x, context):
    parts = x.chunk(context.group.size(), dim=2)
    return all_to_all(context.group, torch.cat(parts, dim=1).squeeze(0)).unsqueeze(0)


def _to_sequence_parallel(x, context):
    exchanged = all_to_all(context.group, x.squeeze(0)).unsqueeze(0)
    return torch.cat(exchanged.chunk(context.group.size(), dim=1), dim=2).contiguous()


def _head_slice(x, context):
    return x.chunk(context.group.size(), dim=0)[context.group.rank()].contiguous()


def build_cp_context(cu_seqlens, group, conv1d_kernel_size):
    return SimpleNamespace(cu_seqlens=cu_seqlens, group=group)


def _convolve(x, weight, activation, cu):
    # Dispatch Conv1d independently of KDA availability. The torch backend
    # can be selected explicitly through the environment setting.
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
    if _can_shard_heads(x, cp_context):
        out = _convolve(
            _to_head_parallel(x, cp_context), _head_slice(weight, cp_context), activation, cp_context.cu_seqlens
        )
        return _to_sequence_parallel(out, cp_context), None
    cu_seqlens = cp_context.cu_seqlens if cp_context is not None else None
    out = _convolve(_gather(x, cp_context), weight, activation, cu_seqlens)
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

    shard_heads = _can_shard_heads(q, cp_context)
    if shard_heads:
        heads = q.shape[2]
        if kwargs.get("A_log") is not None:
            kwargs["A_log"] = _head_slice(kwargs["A_log"], cp_context)
        if kwargs.get("dt_bias") is not None:
            kwargs["dt_bias"] = _head_slice(kwargs["dt_bias"].reshape(heads, -1), cp_context).flatten()
        q, k, v, g, beta = [_to_head_parallel(x, cp_context) for x in (q, k, v, g, beta)]
    else:
        q, k, v, g, beta = [_gather(x, cp_context).contiguous() for x in (q, k, v, g, beta)]

    def local_output(out):
        return _to_sequence_parallel(out, cp_context) if shard_heads else _local(out, cp_context)

    cpu_bounds = kwargs.pop("cu_seqlens_cpu", None)
    bounds = _boundaries(cpu_bounds if cpu_bounds is not None else cu_seqlens, q.shape[1])
    mode = os.getenv("GLM53_KDA_SEQUENCE_MODE", "loop")
    if mode not in ("loop", "packed"):
        raise ValueError(f"Unsupported GLM53_KDA_SEQUENCE_MODE: {mode}")
    if mode == "packed":
        # Remove empty sequences without changing token positions.
        bounds = list(dict.fromkeys(bounds))
        with torch.profiler.record_function("glm53::kda_packed"):
            out, _ = chunk_kda_ascendc(q, k, v, g=g, beta=beta, cu_seqlens=bounds, **kwargs)
        return local_output(out), None
    outputs = []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        if start == end:
            continue
        out, _ = chunk_kda_ascendc(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            **kwargs,
        )
        outputs.append(out)
    return local_output(torch.cat(outputs, dim=1)), None


class SparseMLA:
    @staticmethod
    def apply(q, kv, indices, scaling):
        # Keep CANN SMLA loading lazy for rollout/driver-only imports.
        from .ascendc_smla import SparseMLA as AscendCSparseMLA

        return AscendCSparseMLA.apply(q, kv, indices, scaling)
