# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0.
"""A5 SparseFlashMla adapter for GLM-5.3-Flash's shared absorbed KV."""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_smla_ops():
    """Load the official CANN transformer extension only when SMLA is selected."""
    try:
        import cann_ops_transformer
    except ImportError as exc:
        raise RuntimeError(
            "SparseFlashMla requires a cann_ops_transformer package that provides "
            "sparse_flash_mla and sparse_flash_mla_grad."
        ) from exc

    required = (
        "sparse_flash_mla_metadata",
        "sparse_flash_mla",
        "sparse_flash_mla_grad_metadata",
        "sparse_flash_mla_grad",
    )
    namespace = cann_ops_transformer
    if not all(hasattr(namespace, name) for name in required):
        namespace = getattr(cann_ops_transformer, "ops", None)
    missing = [name for name in required if namespace is None or not hasattr(namespace, name)]
    if missing:
        raise RuntimeError(f"cann_ops_transformer is missing required SparseFlashMla APIs: {missing}.")
    return namespace


def get_sparse_topk_lengths(indices: torch.Tensor) -> torch.Tensor:
    """Return the valid-prefix length expected by ``ori_topk_length``.

    KPool emits a contiguous valid prefix followed by ``-1`` padding, so the
    length is the index of the first invalid entry (or the full top-k width
    when every entry is valid).  Two parallel reductions replace the previous
    serial ``cumprod``: on Ascend the int32 ``argmax``/``all`` path is about
    2000x faster (0.1 ms vs 227 ms for a [8192, 1, 2048] tensor), while
    malformed entries after the first ``-1`` stay unreachable.
    """
    topk = indices.shape[-1]
    if topk == 0:
        return torch.zeros(indices.shape[:-1], dtype=torch.int32, device=indices.device)
    valid = indices.ge(0)
    # int32 argmax uses the AI Core path; int8 argmax is ~300x slower on Ascend.
    # argmax returns int64, but aclnnSparseFlashMlaMetadata requires int32.
    first_invalid = (~valid).to(torch.int32).argmax(dim=-1).to(torch.int32)
    return first_invalid.masked_fill(valid.all(dim=-1), topk)


def _canonical_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize canonical strides, including otherwise-contiguous size-1 axes."""
    expected = []
    stride = 1
    for size in reversed(tensor.shape):
        expected.append(stride)
        stride *= size
    if tuple(tensor.stride()) == tuple(reversed(expected)):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _get_tnd_cu_seqlens(packed_seq_params, *, kv: bool) -> torch.Tensor:
    suffix = "kv" if kv else "q"
    padded = getattr(packed_seq_params, f"cu_seqlens_{suffix}_padded", None)
    result = padded if padded is not None else getattr(packed_seq_params, f"cu_seqlens_{suffix}", None)
    if result is None or result.ndim != 1 or result.numel() < 2:
        raise ValueError(f"SMLA TND requires a boundary-format cu_seqlens_{suffix} tensor.")
    return result.to(dtype=torch.int32).contiguous()


def _build_metadata(
    metadata_op,
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    ori_sparse_indices: torch.Tensor,
    ori_topk_length: torch.Tensor,
    layout: str,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_ori_kv: torch.Tensor | None,
) -> torch.Tensor:
    if layout == "BSND":
        batch_size = q.size(0)
        max_seqlen_q = q.size(1)
        max_seqlen_ori_kv = ori_kv.size(1)
    else:
        batch_size = None
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
        max_seqlen_ori_kv = int((cu_seqlens_ori_kv[1:] - cu_seqlens_ori_kv[:-1]).max().item())
    return metadata_op(
        q.size(-2),
        ori_kv.size(-2),
        q.size(-1),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        ori_topk_length=ori_topk_length,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_ori_kv=max_seqlen_ori_kv,
        max_seqlen_cmp_kv=0,
        ori_topk=ori_sparse_indices.size(-1),
        cmp_topk=0,
        cmp_ratio=1,
        ori_mask_mode=0,
        cmp_mask_mode=0,
        ori_win_left=-1,
        ori_win_right=-1,
        layout_q=layout,
        layout_kv=layout,
        has_ori_kv=True,
        has_cmp_kv=False,
    )


def _operator_kwargs(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    ori_sparse_indices: torch.Tensor,
    ori_topk_length: torch.Tensor,
    metadata: torch.Tensor,
    softmax_scale: float,
    layout: str,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_ori_kv: torch.Tensor | None,
) -> dict:
    return {
        "ori_kv": ori_kv,
        "ori_sparse_indices": ori_sparse_indices,
        # The 1014 A5 kernel requires this input.  -inf keeps it invisible
        # while GLM remains a model without attention sinks.
        "sinks": torch.full((q.size(-2),), float("-inf"), dtype=torch.float32, device=q.device),
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_ori_kv": cu_seqlens_ori_kv,
        "ori_topk_length": ori_topk_length,
        "metadata": metadata,
        "softmax_scale": softmax_scale,
        "cmp_ratio": 1,
        "ori_mask_mode": 0,
        "cmp_mask_mode": 0,
        "ori_win_left": -1,
        "ori_win_right": -1,
        "layout_q": layout,
        "layout_kv": layout,
    }


class _SparseFlashMlaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        ori_kv,
        ori_sparse_indices,
        ori_topk_length,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        softmax_scale,
        layout,
    ):
        ops = _load_smla_ops()
        metadata = _build_metadata(
            ops.sparse_flash_mla_metadata,
            q,
            ori_kv,
            ori_sparse_indices,
            ori_topk_length,
            layout,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
        )
        operator_args = _operator_kwargs(
            q,
            ori_kv,
            ori_sparse_indices,
            ori_topk_length,
            metadata,
            softmax_scale,
            layout,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
        )
        attn_out, softmax_lse = ops.sparse_flash_mla(
            q,
            ori_block_table=None,
            cmp_block_table=None,
            topk_value_mode=1,
            return_softmax_lse=True,
            **operator_args,
        )
        ctx.save_for_backward(q, ori_kv, ori_sparse_indices, ori_topk_length, attn_out, softmax_lse)
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_ori_kv = cu_seqlens_ori_kv
        ctx.softmax_scale = softmax_scale
        ctx.layout = layout
        return attn_out

    @staticmethod
    def backward(ctx, dout):
        q, ori_kv, ori_sparse_indices, ori_topk_length, attn_out, softmax_lse = ctx.saved_tensors
        ops = _load_smla_ops()
        metadata = _build_metadata(
            ops.sparse_flash_mla_grad_metadata,
            q,
            ori_kv,
            ori_sparse_indices,
            ori_topk_length,
            ctx.layout,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_ori_kv,
        )
        operator_args = _operator_kwargs(
            q,
            ori_kv,
            ori_sparse_indices,
            ori_topk_length,
            metadata,
            ctx.softmax_scale,
            ctx.layout,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_ori_kv,
        )
        dq, dori_kv, *_ = ops.sparse_flash_mla_grad(
            q,
            dout.contiguous(),
            attn_out,
            softmax_lse,
            **operator_args,
        )
        return dq, dori_kv, None, None, None, None, None, None


def fused_sparse_flash_mla(
    query: torch.Tensor,
    ori_kv: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    *,
    packed_seq_params=None,
) -> torch.Tensor:
    """Run SMLA while preserving KPool's trailing ``-1`` indices."""
    is_tnd = packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd"
    if is_tnd:
        if query.ndim != 3 or ori_kv.ndim != 3 or topk_indices.ndim != 3 or topk_indices.size(1) != 1:
            raise ValueError("SMLA TND expects q/kv [T,N,D] and indices [T,1,K].")
        q = _canonical_contiguous(query)
        kv = _canonical_contiguous(ori_kv)
        indices = topk_indices.to(dtype=torch.int32).contiguous()
        cu_q = _get_tnd_cu_seqlens(packed_seq_params, kv=False)
        cu_kv = _get_tnd_cu_seqlens(packed_seq_params, kv=True)
        layout = "TND"
    else:
        if query.ndim != 4 or ori_kv.ndim != 4 or topk_indices.ndim != 3:
            raise ValueError("SMLA BSND adapter expects q/kv [S,B,N,D] and indices [B,S,K].")
        q = _canonical_contiguous(query.permute(1, 0, 2, 3))
        kv = _canonical_contiguous(ori_kv.permute(1, 0, 2, 3))
        indices = topk_indices.unsqueeze(2).to(dtype=torch.int32).contiguous()
        cu_q = None
        cu_kv = None
        layout = "BSND"
    if q.size(-1) != 512 or kv.size(-1) != 512:
        raise ValueError(f"A5 SparseFlashMla requires head_dim=512, got q={q.size(-1)}, kv={kv.size(-1)}.")
    if kv.size(-2) != 1:
        raise ValueError(f"A5 SparseFlashMla requires one shared KV head, got {kv.size(-2)}.")
    lengths = get_sparse_topk_lengths(indices)
    output = _SparseFlashMlaFunction.apply(q, kv, indices, lengths, cu_q, cu_kv, softmax_scale, layout)
    return output if is_tnd else output.permute(1, 0, 2, 3).contiguous()
