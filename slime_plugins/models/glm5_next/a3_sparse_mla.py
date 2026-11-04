"""A3 sparse-MLA boundary using CANN SparseFlashAttention forward/backward.

GLM's absorbed KV is both key and value. Zero RoPE channels preserve its
no-RoPE scores; trailing -1 index padding preserves the sparse selection.
"""
import torch
import torch.nn.functional as F
import torch_npu


class _A3SparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
            raise ValueError('A3 sparse MLA expects q/kv [T,N,D], indices [T,1,K]')
        if q.shape[-1] != 512 or kv.shape[-2:] != (1, 512):
            raise ValueError('A3 sparse MLA requires 512 channels and one shared KV head')
        if indices.shape[:2] != (q.shape[0], 1):
            raise ValueError('Sparse indices must provide one shared selection per query')
        width = ((indices.shape[-1] + 1023) // 1024) * 1024
        if not 1024 <= width <= 8192:
            raise ValueError('A3 sparse attention supports padded selection widths 1024..8192')
        selected = F.pad(indices.to(torch.int32), (0, width - indices.shape[-1]), value=-1)
        nonempty = selected[..., :1] >= 0
        # Give empty/padding queries a harmless key, then mask their output and
        # upstream gradient. This avoids A3's zero-selection scatter path while
        # preserving exactly zero attention and gradients for empty selections.
        first = torch.where(nonempty, selected[..., :1], 0)
        selected = torch.cat((first, selected[..., 1:]), dim=-1)
        query = q.unsqueeze(0).contiguous()
        # A3's dense-sequence shortcut assumes selected_count >= KV length
        # implies a complete, unique selection. KPool may retain duplicates.
        # Append unselected KV rows so the sparse path is always used; indices
        # still reference only the original KV, so scores/gradients are unchanged.
        kv_padding = max(0, width + 1 - kv.shape[0])
        padded_kv = F.pad(kv, (0, 0, 0, 0, 0, kv_padding)) if kv_padding else kv
        shared = padded_kv.unsqueeze(0).contiguous()
        selected = selected.unsqueeze(0).contiguous()
        query_rope = query.new_zeros((*query.shape[:-1], 64))
        key_rope = shared.new_zeros((*shared.shape[:-1], 64))
        output, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
            query, shared, shared, selected, scaling,
            query_rope=query_rope, key_rope=key_rope,
            sparse_block_size=1, layout_query='BSND', layout_kv='BSND',
            sparse_mode=0, attention_mode=2, return_softmax_lse=True)
        ctx.save_for_backward(query, shared, selected, query_rope, key_rope,
                              output, softmax_max, softmax_sum, nonempty.unsqueeze(0))
        ctx.scaling = scaling
        ctx.kv_tokens = kv.shape[0]
        return torch.where(nonempty, output.squeeze(0), 0)

    @staticmethod
    def backward(ctx, dout):
        query, shared, selected, _, _, _, _, _, nonempty = ctx.saved_tensors
        # The shared A3 implementation uses this analytical FP32 derivative
        # for long/unequal Q/KV selections: native SFAG did not pass its
        # sparse-index gradient reference. Keep each original selected slot,
        # including multiplicity, and bound live score memory to 32 queries.
        # Only this A3 boundary changes; private training code stays pinned.
        key = shared[0, :, 0].float()
        dq = torch.empty_like(query)
        dkv = torch.zeros_like(key)
        for first in range(0, query.shape[1], 32):
            last = min(first + 32, query.shape[1])
            q = query[0, first:last].float().transpose(0, 1)
            dy = torch.where(nonempty[0, first:last], dout[first:last], 0).float().transpose(0, 1)
            indices = selected[0, first:last, 0]
            valid = indices >= 0
            counts = torch.zeros((last-first, key.shape[0]), device=query.device, dtype=torch.float32)
            counts.scatter_add_(1, indices.clamp_min(0).long(), valid.float())
            keep = counts > 0
            scores = (q @ key.T) * ctx.scaling + counts.clamp_min(1).log()
            scores.masked_fill_(~keep, -float('inf'))
            probability = scores.softmax(-1)
            dp = dy @ key.T
            ds = probability * (dp - (probability * dp).sum(-1, keepdim=True)) * ctx.scaling
            dq[0, first:last] = (ds @ key).transpose(0, 1).to(query.dtype)
            dkv.add_((ds.transpose(-1, -2) @ q + probability.transpose(-1, -2) @ dy).sum(0))
        return dq.squeeze(0), dkv[:ctx.kv_tokens].unsqueeze(1).to(shared.dtype), None, None


def sparse_mla(q, kv, indices, scaling):
    return _A3SparseMLA.apply(q, kv, indices, scaling)
