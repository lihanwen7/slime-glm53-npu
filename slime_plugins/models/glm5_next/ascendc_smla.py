"""Slime GLM5-next no-RoPE boundary for MindSpeed-Bridge's unchanged SMLA.

The caller supplies absorbed 512-channel queries/KV and global sparse indices.
The generic GLM5/RoPE SparseMLA interface is deliberately left untouched.
"""
import torch
from . import mindspeed_smla
from .cann_smla_ops import load_sparse_ops

# The SDK's top-level import eagerly builds unrelated operators and imports
# their graph converters. Keep the upstream math/autograd file byte-for-byte;
# replace only its lazy package loader with the two original SDK modules.
mindspeed_smla._load_smla_ops = load_sparse_ops
fused_sparse_flash_mla = mindspeed_smla.fused_sparse_flash_mla


def compact_sparse_indices(indices, kv_length):
    """Move valid slots to a prefix, retaining their order and multiplicity."""
    indices = indices.to(dtype=torch.int32)
    valid = (indices >= 0) & (indices < kv_length)
    position = torch.arange(indices.size(-1), device=indices.device, dtype=torch.int32)
    # Unique position keys preserve order without relying on stable-sort support.
    keys = torch.where(valid, position, position + indices.size(-1))
    # Sort position keys in FP32; sparse index values remain int32.
    # The current width is 2112, so every key in [0, 4223] is exact.
    keys = keys.float()
    order = keys.argsort(-1)
    return torch.where(valid, indices, -1).gather(-1, order).contiguous()


class SparseMLA:
    @staticmethod
    def apply(q, kv, indices, scaling):
        if kv.size(1) != 1 or indices.size(1) != 1:
            raise ValueError("GLM5-next SparseMLA requires one shared KV/index head")
        # The NPU call site passes its unpadded no-RoPE tensors. The unchanged
        # Bridge implementation checks the native 512-channel shape itself.
        q = q.contiguous()
        kv = kv.contiguous()
        selected = compact_sparse_indices(indices, kv.shape[0])
        if torch.npu.get_device_name(q.device.index).startswith("Ascend910_93"):
            from .a3_sparse_mla import sparse_mla

            return sparse_mla(q, kv, selected, scaling), None
        # Bridge takes SBND, then canonicalizes to the native B=1 BSND layout.
        out = fused_sparse_flash_mla(
            q.unsqueeze(0).transpose(0, 1),
            kv.unsqueeze(0).transpose(0, 1),
            selected[:, 0].unsqueeze(0),
            scaling,
        )
        # The GLM5-next caller discards its second return value. Native ln-LSE
        # remains inside the upstream autograd function for its own backward.
        return out.squeeze(1), None
