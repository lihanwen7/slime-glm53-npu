"""AscendC causal Conv1d forward/backward for GLM5-next training.

Adapted from MindSpeed-Bridge's GLM-5.3-Flash kda_conv1d_fused.py.
Inputs retain Slime's [B, T, C] and [C, W] contract. SiLU stays outside
the convolution, as in the reference adapter. CP gather/scatter is owned
by npu_backend, so packed boundaries here describe complete sequences.
"""

import logging
import ctypes
import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
_LOGGED = False
_BACKWARD_LOGGED = False


@lru_cache(maxsize=1)
def _load_ops():
    from fla_npu.ops.ascendc import npu_causal_conv1d, npu_causal_conv1d_bwd
    from fla_npu.ops.ascendc._runtime import runtime

    acl = runtime()
    symbols = ("aclnnCausalConv1d", "aclnnCausalConv1dGetWorkspaceSize",
               "aclnnCausalConv1dBwd", "aclnnCausalConv1dBwdGetWorkspaceSize")
    try:
        for name in symbols:
            acl.symbol(name)
    except AttributeError:
        # Older FLA wheels embed KDA but omit Conv1d. Extend only their missing
        # convolution symbols; never replace their KDA implementations.
        root = Path(__file__).resolve().parents[5]
        vendor = root / "cann-custom-ops/glm53-conv/vendors/glm53_conv_transformer"
        library = ctypes.CDLL(str(vendor / "op_api/lib/libcust_opapi.so"),
                              mode=os.RTLD_LOCAL | os.RTLD_NOW | os.RTLD_NODELETE)
        resolved = {name: getattr(library, name) for name in symbols}
        acl._symbols.update(resolved)
        # ctypes function objects retain the library; cache it explicitly too.
        return npu_causal_conv1d, npu_causal_conv1d_bwd, library
    return npu_causal_conv1d, npu_causal_conv1d_bwd, None


class _CausalConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, boundaries):
        npu_causal_conv1d, _, _ = _load_ops()

        global _LOGGED
        if not _LOGGED:
            logger.info("GLM53_FUSED_CONV backend=ascendc forward_and_backward=True")
            _LOGGED = True
        width, channels = weight.shape
        sequences = len(boundaries) - 1 if boundaries is not None else x.shape[0]
        states = x.new_zeros((sequences, width - 1, channels))
        with torch.profiler.record_function("glm53::ascendc_causal_conv1d_forward"):
            output = npu_causal_conv1d(
                x=x, weight=weight, bias=None, conv_states=states,
                query_start_loc=boundaries, activation_mode=0, run_mode=0,
            )
        ctx.save_for_backward(x, weight)
        ctx.boundaries = boundaries
        # FLA 26.10 uses BSND for logical [B, T, C] Conv1d inputs;
        # this enum does not require reshaping them to four dimensions.
        ctx.layout = "TND" if x.ndim == 2 else "BSND"
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _, npu_causal_conv1d_bwd, _ = _load_ops()
        global _BACKWARD_LOGGED
        if not _BACKWARD_LOGGED:
            logger.info("GLM53_FUSED_CONV_BACKWARD backend=ascendc")
            _BACKWARD_LOGGED = True

        x, weight = ctx.saved_tensors
        with torch.profiler.record_function("glm53::ascendc_causal_conv1d_backward"):
            dx, dw, _, _ = npu_causal_conv1d_bwd(
                x=x, y=None, weight=weight, dy=grad_output.contiguous(),
                initial_state=None, dht=None, query_start_loc=ctx.boundaries,
                activation=0, input_layout=ctx.layout,
            )
        return dx, dw, None


def causal_conv1d_ascendc(x, weight, activation="silu", cu_seqlens=None):
    """Apply packed or dense causal depthwise convolution without recurrent state."""
    if x.ndim != 3 or weight.ndim != 2 or x.shape[-1] != weight.shape[0]:
        raise ValueError("Expected x=[B,T,C] and weight=[C,W]")
    if activation not in (None, "silu"):
        raise ValueError(f"Unsupported convolution activation: {activation}")
    if x.device.type != "npu" or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("AscendC Conv1d requires BF16/FP16 NPU inputs")
    boundaries = None
    logical_x = x.contiguous()
    if cu_seqlens is not None:
        if x.shape[0] != 1:
            raise ValueError("Packed AscendC Conv1d requires batch size 1")
        values = [int(v) for v in cu_seqlens.tolist()]
        if len(values) < 2 or values[0] != 0 or values[-1] != x.shape[1]:
            raise ValueError("Packed boundaries must cover all input tokens")
        if any(b < a for a, b in zip(values, values[1:])):
            raise ValueError("Packed boundaries must be nondecreasing")
        # Empty packed sequences have no output or gradient; the kernel requires
        # strictly increasing boundaries.
        boundaries = [values[0]] + [b for a, b in zip(values, values[1:]) if b > a]
        logical_x = logical_x.squeeze(0)
    logical_weight = weight.transpose(0, 1).contiguous().to(x.dtype)
    output = _CausalConv1d.apply(logical_x, logical_weight, boundaries)
    if activation == "silu":
        output = F.silu(output)
    if output.ndim == 2:
        output = output.unsqueeze(0)
    return output
