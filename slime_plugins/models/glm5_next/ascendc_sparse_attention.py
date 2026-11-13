"""SparseFlashAttention autograd for GLM5-next's absorbed, shared KV.

The sparse indices already encode causality and packed-sequence boundaries.
Use sparse_mode=0 in both directions, preserving duplicate selected slots.
CANN 9.2 KV-merge (value=None) returns dK+dV in d_key and an empty d_value.
A3 requires 64 auxiliary RoPE channels. Real keys have zero RoPE, preserving
no-RoPE model scores. Padding maps to a pool of zero KVs with large negative auxiliary scores,
because the A3 backward kernel cannot consume -1 indices. Distributing padding
indices avoids the measured slowdown from repeatedly selecting one sentinel.
"""
import math

import torch
import torch.nn.functional as F
import torch_npu


# 512 masked KV rows avoid the repeated-sentinel slowdown on A3 while adding
# only 512 KiB of BF16/FP16 KV storage (plus auxiliary RoPE storage).
_PADDING_KV_ROWS = 512


class _SparseFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
            raise ValueError("SparseFlashAttention expects Q/KV [T,N,D], indices [T,1,K]")
        if q.shape[-1] != 512 or kv.shape[-2:] != (1, 512):
            raise ValueError("SparseFlashAttention requires 512 channels and one shared KV head")
        if not q.shape[0] or not kv.shape[0] or indices.shape[:2] != (q.shape[0], 1) or not indices.shape[-1]:
            raise ValueError("SparseFlashAttention requires nonempty Q/KV and selections [T,1,K]")
        if not math.isfinite(float(scaling)) or scaling <= 0:
            raise ValueError("SparseFlashAttention requires a positive finite softmax scale")
        if q.dtype not in (torch.bfloat16, torch.float16) or kv.dtype != q.dtype:
            raise ValueError("SparseFlashAttention requires matching BF16 or FP16 Q/KV")
        with torch.npu.device(q.device):
            query = q.contiguous()
            selected = indices.to(torch.int32).contiguous()
            nonempty = selected[..., :1].ge(0)
            selected = torch.cat((torch.where(nonempty, selected[..., :1], 0), selected[..., 1:]), -1)
            # S2 must be strictly greater than K: otherwise A3 backward selects
            # a dense-prefix fast path that ignores nonmonotonic sparse indices.
            # Reserve a pool of masked padding rows, never real tokens.
            tokens = max(q.shape[0], kv.shape[0] + _PADDING_KV_ROWS,
                         selected.shape[-1] + 1)
            shared = F.pad(kv, (0, 0, 0, 0, 0, tokens-kv.shape[0])).contiguous()
            q_rope = query.new_zeros((*query.shape[:-1], 64))
            k_rope = shared.new_zeros((*shared.shape[:-1], 64))
            # Avoid invalid GM reads in SFAG for -1 padding. Real key scores
            # remain unchanged; the sentinel probability underflows to zero.
            q_rope[..., 0] = 1
            k_rope[-_PADDING_KV_ROWS:, :, 0] = torch.finfo(kv.dtype).min
            padding_indices = (
                torch.arange(selected.shape[-1], device=selected.device, dtype=torch.int32)
                % _PADDING_KV_ROWS + tokens - _PADDING_KV_ROWS
            )
            selected = torch.where(selected < 0, padding_indices, selected)
            q_lengths = torch.tensor([query.shape[0]], dtype=torch.int32, device=q.device)
            kv_lengths = torch.tensor([tokens], dtype=torch.int32, device=q.device)
            output, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
                query, shared, None, selected, float(scaling),
                query_rope=q_rope, key_rope=k_rope,
                actual_seq_lengths_query=q_lengths, actual_seq_lengths_kv=kv_lengths,
                sparse_block_size=1, layout_query="TND", layout_kv="TND",
                sparse_mode=0, attention_mode=2, return_softmax_lse=True,
            )
            ctx.save_for_backward(query, shared, selected, output, softmax_max, softmax_sum,
                                  q_rope, k_rope, q_lengths, kv_lengths, nonempty)
            ctx.scaling = float(scaling)
            ctx.kv_tokens = kv.shape[0]
            return torch.where(nonempty, output, 0)

    @staticmethod
    def backward(ctx, dout):
        q, kv, selected, output, maximum, total, qr, kr, ql, kl, nonempty = ctx.saved_tensors
        with torch.npu.device(q.device):
            upstream = torch.where(nonempty, dout, 0).contiguous()
            dq, dkv, dv, *_ = torch_npu.npu_sparse_flash_attention_grad(
                q, kv, None, selected, upstream, output, maximum, total, ctx.scaling, 1,
                query_rope=qr, key_rope=kr, actual_seq_qlen=ql, actual_seq_kvlen=kl,
                layout="TND", sparse_mode=0, attention_mode=2,
            )
            if dv.numel() != 0:
                raise RuntimeError("SparseFlashAttention KV-merge must return an empty d_value")
            return dq, dkv[:ctx.kv_tokens], None, None


def sparse_attention(q, kv, indices, scaling):
    """Indices must contain valid global KV offsets followed by -1 padding."""
    return _SparseFlashAttention.apply(q, kv, indices, scaling)
