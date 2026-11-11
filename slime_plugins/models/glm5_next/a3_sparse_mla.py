"""A3 SparseFlashMla autograd boundary for shared absorbed KV.

Both directions call cann_ops_transformer.ops. Only the A3 forward needs
paged KV (PA_BBND); backward consumes contiguous BSND and natural-log LSE.
"""
import torch
import torch.nn.functional as F

from .cann_smla_ops import load_sparse_ops


def _lengths(indices):
    valid = indices.ge(0)
    first = (~valid).to(torch.int32).argmax(dim=-1).to(torch.int32)
    return first.masked_fill(valid.all(dim=-1), indices.shape[-1])


def _metadata(op, q, kv, selected, lengths):
    return op(
        q.shape[-2], 1, 512,
        cu_seqlens_q=None, cu_seqlens_ori_kv=None,
        ori_topk_length=lengths, batch_size=1,
        max_seqlen_q=q.shape[1], max_seqlen_ori_kv=kv.shape[1],
        max_seqlen_cmp_kv=0, ori_topk=selected.shape[-1], cmp_topk=0,
        cmp_ratio=1, ori_mask_mode=0, cmp_mask_mode=0,
        ori_win_left=0, ori_win_right=0,
        layout_q="BSND", layout_kv="BSND", has_ori_kv=True, has_cmp_kv=False,
    )


def _arguments(q, kv, selected, lengths, metadata, scale):
    return dict(
        ori_kv=kv, ori_sparse_indices=selected, ori_topk_length=lengths,
        sinks=torch.full((q.shape[-2],), -float("inf"),
                         dtype=torch.float32, device=q.device),
        cu_seqlens_q=None, cu_seqlens_ori_kv=None,
        metadata=metadata, softmax_scale=scale, cmp_ratio=1,
        ori_mask_mode=0, cmp_mask_mode=0, ori_win_left=0, ori_win_right=0,
        layout_q="BSND", layout_kv="BSND",
    )


class _A3SparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
            raise ValueError("A3 MLA expects q/kv [T,N,D], indices [T,1,K]")
        if q.shape[-1] != 512 or kv.shape[-2:] != (1, 512):
            raise ValueError("A3 MLA requires 512 channels and one shared KV head")
        if indices.shape[:2] != (q.shape[0], 1) or indices.shape[-1] == 0:
            raise ValueError("A3 MLA requires a nonzero selection width per query")
        if not q.shape[0] or not kv.shape[0]:
            raise ValueError("A3 MLA requires nonempty Q and KV")
        ops = load_sparse_ops()
        with torch.npu.device(q.device):
            selected = indices.to(torch.int32).contiguous()
            nonempty = selected[..., :1].ge(0)
            # Empty selections have zero output and gradients. Use a safe native
            # selection, mask the returned output and its upstream gradient.
            selected = torch.cat((torch.where(nonempty, selected[..., :1], 0),
                                  selected[..., 1:]), dim=-1).unsqueeze(0)
            lengths = _lengths(selected)
            query = q.reshape(1, *q.shape).contiguous()
            # PA_BBND requires blocks of 16. Keeping S2 >= selection width also
            # avoids the backward length cap truncating repeated selected slots.
            # S2 >= S1 also bypasses A3 dense-prefix pruning for unequal Q/KV.
            padded_tokens = ((max(q.shape[0], kv.shape[0], selected.shape[-1]) + 15) // 16) * 16
            shared = F.pad(kv, (0, 0, 0, 0, 0, padded_tokens-kv.shape[0]))
            shared = shared.reshape(1, padded_tokens, 1, 512).contiguous()
            args = _arguments(query, shared, selected, lengths, None, scaling)
            args.update(
                ori_kv=shared.reshape(-1, 16, 1, 512),
                ori_sparse_indices=selected.reshape(q.shape[0], 1, -1),
                ori_topk_length=lengths.reshape(q.shape[0], 1),
                ori_block_table=torch.arange(padded_tokens//16, dtype=torch.int32,
                                             device=q.device).reshape(1, -1),
                cmp_block_table=None,
                cu_seqlens_q=torch.tensor([0, q.shape[0]], dtype=torch.int32,
                                         device=q.device),
                seqused_ori_kv=torch.tensor([padded_tokens], dtype=torch.int32,
                                           device=q.device),
                layout_q="TND", layout_kv="PA_BBND",
                topk_value_mode=1, return_softmax_lse=True,
            )
            args["metadata"] = ops.sparse_flash_mla_metadata(
                q.shape[-2], 1, 512,
                cu_seqlens_q=args["cu_seqlens_q"],
                seqused_ori_kv=args["seqused_ori_kv"],
                # The installed A3 TND kernel cannot merge S2 shards. Schedule
                # whole queries by estimating one KV slot per query; the actual
                # MLA call below still receives every real selected slot.
                ori_topk_length=torch.ones_like(args["ori_topk_length"]),
                batch_size=1, max_seqlen_q=q.shape[0],
                max_seqlen_ori_kv=padded_tokens, max_seqlen_cmp_kv=0,
                ori_topk=selected.shape[-1], cmp_topk=0, cmp_ratio=1,
                ori_mask_mode=0, cmp_mask_mode=0, ori_win_left=0, ori_win_right=0,
                layout_q="TND", layout_kv="PA_BBND",
                has_ori_kv=True, has_cmp_kv=False,
            )
            output, lse = ops.sparse_flash_mla(query[0], **args)
            output = output.reshape_as(query)
            lse = lse.reshape(1, 1, q.shape[0], q.shape[1])
            ctx.save_for_backward(query, shared, selected, lengths, output, lse,
                                  nonempty)
            ctx.scaling = scaling
            ctx.kv_tokens = kv.shape[0]
            return torch.where(nonempty, output[0], 0)

    @staticmethod
    def backward(ctx, dout):
        q, kv, selected, lengths, output, lse, nonempty = ctx.saved_tensors
        ops = load_sparse_ops()
        with torch.npu.device(q.device):
            metadata = _metadata(ops.sparse_flash_mla_grad_metadata, q, kv,
                                 selected, lengths)
            args = _arguments(q, kv, selected, lengths, metadata, ctx.scaling)
            upstream = torch.where(nonempty, dout, 0).unsqueeze(0).contiguous()
            dq, dkv, *_ = ops.sparse_flash_mla_grad(q, upstream, output, lse, **args)
            return dq[0], dkv[0, :ctx.kv_tokens], None, None


def sparse_mla(q, kv, indices, scaling):
    return _A3SparseMLA.apply(q, kv, indices, scaling)
