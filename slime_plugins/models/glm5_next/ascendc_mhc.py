"""GLM5-next mHC adapters for the CANN operators used by MindSpeed-Bridge."""
import logging
from functools import lru_cache

import torch

from .cann_smla_ops import load_mhc_ops

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _log(part):
    logger.info("GLM53_FUSED_MHC part=%s backend=cann_ascendc", part)


@lru_cache(maxsize=1)
def _mhc_pre_autograd(pre):
    """Adapt CANN 9.2's residual gradient in this caller, leaving the SDK intact.

    Forward exposes [..., N*N], but the native backward expects [..., N, N].
    Reuse the SDK's forward, saved tensors and backward; only normalize that
    gradient before handing it to the SDK. Reshaping the forward output alone
    cannot fix this: autograd reshapes its gradient back to the flat output.
    """
    sdk_function = pre.__globals__.get("MhcPreSinkhornFunction")
    if sdk_function is None:
        raise RuntimeError("Expected the pinned CANN 9.2 mHC autograd interface")

    class MhcPreSinkhornLayoutFunction(sdk_function):
        @staticmethod
        def backward(ctx, grad_hin, grad_h_post, grad_h_res):
            x = ctx.saved_tensors[0]
            n = x.shape[-2]
            if grad_h_res.shape == (*x.shape[:-2], n * n):
                grad_h_res = grad_h_res.reshape(*x.shape[:-2], n, n).contiguous()
            return sdk_function.backward(ctx, grad_hin, grad_h_post, grad_h_res)

    return MhcPreSinkhornLayoutFunction


def mhc_pre_ascendc(x, weight, scale, bias, n, norm_eps, eps, iterations):
    if n != 4 or iterations != 20:
        raise ValueError("CANN fused mHC requires 4 streams and 20 Sinkhorn iterations")
    if x.shape[-1] % n:
        raise ValueError("mHC stream width must divide the hidden dimension")
    hidden = x.shape[-1] // n
    streams = x.reshape(1, -1, n, hidden).contiguous()
    _log("pre")
    with torch.profiler.record_function("glm53::mhc_pre_sinkhorn"):
        args = (
            streams, weight.float().contiguous(), scale.float().contiguous(),
            bias.float().contiguous(), n, iterations, eps, norm_eps,
        )
        pre = load_mhc_ops().mhc_pre_sinkhorn
        if torch.is_grad_enabled() and any(t.requires_grad for t in args[:4]):
            collapsed, post, residual = _mhc_pre_autograd(pre).apply(*args)
        else:
            collapsed, post, residual = pre(*args)
    return (collapsed.reshape(*x.shape[:-1], hidden),
            post.reshape(*x.shape[:-1], n),
            residual.reshape(*x.shape[:-1], n, n))


def mhc_post_ascendc(y, x, post, residual):
    n = post.shape[-1]
    hidden = y.shape[-1]
    if n != 4:
        raise ValueError("CANN fused mHC requires 4 streams")
    _log("post")
    with torch.profiler.record_function("glm53::mhc_post"):
        result = load_mhc_ops().mhc_post(
            x.reshape(1, -1, n, hidden).contiguous(),
            residual.reshape(1, -1, n, n).float().contiguous(),
            y.reshape(1, -1, hidden).contiguous(),
            post.reshape(1, -1, n).float().contiguous(),
        )
    if result.numel() != x.numel():
        raise RuntimeError("CANN mHC post returned an incompatible output")
    # Pipeline schedules release output storage and require a viewless tensor.
    # clone preserves autograd while removing the reshape view.
    return result.reshape_as(x).clone()
