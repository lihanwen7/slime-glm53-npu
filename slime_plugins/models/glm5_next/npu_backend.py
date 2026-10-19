"""Functional Ascend backend for the native GLM5-next model.

KDA uses the existing AscendC forward/backward bridge. Convolution, gated
normalization and sparse attention use differentiable PyTorch NPU operations.
CP gathers complete packed streams before recurrent operations: this validates
functionality, not FLA's GPU CP performance or fused-kernel performance.
"""

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
    outputs = []
    for start, end in zip((bounds := _boundaries(cu, x.shape[1]))[:-1], bounds[1:], strict=True):
        if start == end:
            continue
        part = x[:, start:end].transpose(1, 2)
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
    bounds = _boundaries(cu_seqlens, q.shape[1])
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
        if kv.shape[1] != 1:
            raise ValueError("GLM5-next NPU sparse MLA expects one compressed KV group")
        q_len, k_len = q.shape[0], kv.shape[0]
        selected = indices[:, 0].long()
        valid = (selected >= 0) & (selected < k_len)
        mask = torch.zeros((q_len, k_len), dtype=torch.bool, device=q.device)
        rows = torch.arange(q_len, device=q.device)[:, None].expand_as(selected)
        mask[rows[valid], selected[valid]] = True
        row_valid = mask.any(-1)
        # Avoid undefined all-masked softmax; remove dummy results and gradients.
        mask[:, 0] |= ~row_valid
        scores = torch.einsum("qhd,kd->hqk", q.float(), kv[:, 0].float()) * scaling
        scores = scores.masked_fill(~mask[None], -torch.inf)
        probabilities = scores.softmax(-1)
        out = torch.einsum("hqk,kd->qhd", probabilities, kv[:, 0, :512].float()).to(q.dtype)
        out = out * row_valid[:, None, None]
        return out, torch.logsumexp(scores, dim=-1).transpose(0, 1)
