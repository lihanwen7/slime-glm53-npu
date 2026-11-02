"""Forward/backward and clamp-boundary checks for the shared fused SwiGLU."""
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
root = Path(__file__).resolve().parents[1] / "slime_plugins/models/glm5_next"
pkg = types.ModuleType("_glm53_swiglu_test")
pkg.__path__ = [str(root)]
sys.modules.setdefault(pkg.__name__, pkg)
ops = importlib.import_module(pkg.__name__ + ".ops")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rows,noncontiguous", [(0, False), (17, False), (512, False), (17, True)])
def test_swiglu_forward_backward(dtype, rows, noncontiguous, monkeypatch):
    monkeypatch.setenv("GLM53_SWIGLU_BACKEND", "ascendc")
    torch.manual_seed(53)
    width, limit = 2048, 10.0
    data = (torch.randn(rows, 2*width, device="npu") * 12).to(dtype)
    if rows:
        edges = torch.tensor([-11., -10., -9., 0., 9., 10., 11.], device="npu", dtype=dtype)
        data[0, :7] = edges
        data[0, width:width+7] = edges
    if noncontiguous:
        data = data.t().contiguous().t()
    x = data.detach().requires_grad_()
    baseline = data.detach().clone().requires_grad_()
    actual = ops.ClampedSwiGLU(SimpleNamespace(activation_func_clamp_value=limit))(x)
    gate, up = baseline.float().chunk(2, dim=-1)
    expected = (F.silu(gate.clamp(max=limit)) * up.clamp(-limit,limit)).to(dtype)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.npu.synchronize()
    for name, a, b in [("forward", actual, expected), ("backward", x.grad, baseline.grad)]:
        assert torch.isfinite(a).all()
        if a.numel():
            diff = (a.float()-b.float())
            rel = (diff.square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-8)).item()
            print(dtype, rows, noncontiguous, name, "max_abs", diff.abs().max().item(),
                  "relative_rms", rel, flush=True)
            assert rel < (1e-5 if dtype == torch.float32 else 1e-3)
    if rows:
        assert x.grad[0, 6].item() == 0  # gate above upper clamp
        assert x.grad[0, width].item() == 0  # up below lower clamp
        assert x.grad[0, width+6].item() == 0  # up above upper clamp
