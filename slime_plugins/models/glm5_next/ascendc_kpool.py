"""Forward-only CANN KPool integration for the frozen GLM5-next indexer.

Based on MindSpeed-Bridge f74d52a8a318c4594202ccdcf96a615b1f2d32da.
Use BSH per document to avoid SDK-dependent packed output capacity layouts.
"""
import logging
from functools import lru_cache

import torch

from .cann_smla_ops import load_kpool_ops

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _log():
    logger.info("GLM53_FUSED_KPOOL backend=cann_ascendc key_pool=True pool_key_indexer=True")


@torch.no_grad()
def key_pool_document(x, wk, gate_weight, ape, norm_weight, norm_bias, norm_eps, pool_size):
    """Compress a single document; discard its incomplete final pool."""
    length = x.shape[0]
    if length < pool_size:
        return x.new_empty((0, wk.shape[0]))
    block_size = min(length, 1024)
    blocks = (length + block_size - 1) // block_size
    cache = torch.zeros((blocks + 1, block_size, 2 * wk.shape[0]),
                        dtype=torch.float32, device=x.device)
    table = torch.arange(1, blocks + 1, dtype=torch.int32, device=x.device).view(1, -1)
    start = torch.zeros(1, dtype=torch.int32, device=x.device)
    with torch.profiler.record_function("glm53::key_pool"):
        pooled = load_kpool_ops().key_pool(
            x.reshape(1, length, -1).contiguous(), wk.contiguous(),
            gate_weight.contiguous(), ape.float().contiguous(), cache, table, start,
            norm_weight=norm_weight.float().contiguous(),
            norm_bias=norm_bias.float().contiguous(), norm_eps=norm_eps,
            cmp_ratio=pool_size,
        )
    return pooled[0, :length // pool_size].contiguous()


@torch.no_grad()
def fused_kpool_indexer(q, x, weights, wk, gate_weight, ape, norm_weight,
                       norm_bias, norm_eps, cu_seqlens, topk, pool_size, *, query_start=0):
    """Return globally indexed causal keys for contiguous allgather-CP queries."""
    if q.shape[-1] != 128 or pool_size not in (2, 4, 8, 16, 32, 64, 128):
        raise ValueError("CANN KPool requires head dimension 128 and a supported power-of-two pool")
    if topk % pool_size:
        raise ValueError("KPool topk must be divisible by pool_size")
    boundaries = cu_seqlens.tolist()
    if boundaries[0] != 0 or boundaries[-1] != x.shape[0]:
        raise ValueError("KPool requires the complete packed hidden-state stream")
    if query_start < 0 or query_start + q.shape[0] > x.shape[0]:
        raise ValueError("KPool query interval lies outside the packed stream")
    width = ((topk + pool_size - 1 + 63) // 64) * 64
    output = torch.full((q.shape[0], 1, width), -1, dtype=torch.int32, device=q.device)
    _log()
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        first = max(start - query_start, 0)
        last = min(end - query_start, q.shape[0])
        if last <= first:
            continue
        # Restrict keys to this rank's final visible prefix. mask_mode=3 then
        # right-aligns local queries with that prefix, including partial pools.
        prefix = query_start + last - start
        pools = prefix // pool_size
        if pools == 0:
            visible = torch.arange(query_start + first - start + 1, prefix + 1,
                                   device=q.device).unsqueeze(1)
            tail = torch.arange(pool_size - 1, device=q.device).unsqueeze(0)
            output[first:last, 0, topk:topk + pool_size - 1] = torch.where(
                tail < visible, tail + start, -1).int()
            continue
        pooled = key_pool_document(x[start:start + prefix], wk, gate_weight, ape,
                                   norm_weight, norm_bias, norm_eps, pool_size)
        with torch.profiler.record_function("glm53::pool_key_indexer"):
            indices, _ = load_kpool_ops().pool_key_indexer(
                q[first:last].unsqueeze(0).contiguous(),
                pooled.reshape(1, pools, 1, -1).contiguous(),
                weights[first:last].reshape(1, last - first, -1).to(q.dtype).contiguous(),
                torch.tensor([prefix % pool_size], dtype=torch.int32, device=q.device),
                layout_q="BSND", layout_k="BSND", topk=topk,
                pool_size=pool_size, mask_mode=3, return_value=False,
            )
        indices = indices.reshape(last - first, -1)
        output[first:last, 0, :topk] = torch.where(indices[:, :topk] >= 0,
                                                  indices[:, :topk] + start, -1).int()
        # Current CANN builds leave the reserved tail slots empty. Preserve
        # GLM's always-select-tail contract explicitly, independently of SDK.
        visible = torch.arange(query_start + first - start + 1, prefix + 1,
                               device=q.device).unsqueeze(1)
        tail = (visible // pool_size) * pool_size + torch.arange(pool_size - 1, device=q.device)
        output[first:last, 0, topk:topk + pool_size - 1] = torch.where(
            tail < visible, tail + start, -1).int()
    return output
