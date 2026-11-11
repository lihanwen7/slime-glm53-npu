"""NPU numerical regression tests for the GLM5-next fused causal convolution."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
_path = Path(__file__).resolve().parents[1] / "slime_plugins/models/glm5_next/ascendc_conv1d.py"
_spec = importlib.util.spec_from_file_location("glm53_conv_test", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def reference(x, weight, activation, boundaries):
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=False):
        if start == end:
            continue
        y = F.conv1d(
            F.pad(x[:, start:end].transpose(1, 2), (weight.shape[-1] - 1, 0)), weight[:, None], groups=weight.shape[0]
        ).transpose(1, 2)
        outputs.append(F.silu(y) if activation else y)
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("case", ["dense", "packed", "short", "empty_segment", "full_width"])
def test_forward_backward(dtype, case):
    torch.manual_seed(53)
    batch, length, channels = (2, 129, 128) if case == "dense" else (1, 257, 128)
    boundaries = [0, length]
    if case == "packed":
        boundaries = [0, 1, 17, 128, length]
    if case == "short":
        length, boundaries = 2, [0, 1, 2]
    if case == "empty_segment":
        boundaries = [0, 0, 37, 37, length]
    if case == "full_width":
        length, channels, boundaries = 1024, 8192, [0, 333, 1024]
    x = (torch.randn(batch, length, channels, device="npu", dtype=dtype) * 0.2).requires_grad_()
    w = (torch.randn(channels, 4, device="npu", dtype=dtype) * 0.2).requires_grad_()
    xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    dy = torch.randn_like(x) * 0.2
    actual = _module.causal_conv1d_ascendc(
        x, w, "silu", None if case == "dense" else torch.tensor(boundaries, dtype=torch.int32)
    )
    expect = reference(xr, wr, "silu", boundaries)
    actual.backward(dy)
    expect.backward(dy)
    torch.npu.synchronize()
    for name, a, b in [("output", actual, expect), ("dx", x.grad, xr.grad), ("dw", w.grad, wr.grad)]:
        af, bf = a.float(), b.float()
        relative_rms = ((af - bf).square().mean().sqrt() / bf.square().mean().sqrt().clamp_min(1e-8)).item()
        print(case, dtype, name, "max_abs", (af - bf).abs().max().item(), "relative_rms", relative_rms, flush=True)
        assert torch.isfinite(a).all()
        assert relative_rms < (0.025 if dtype == torch.bfloat16 else 0.005)


@pytest.mark.parametrize("override,expected", [(None, "ascendc"), ("ascendc", "ascendc"), ("torch", "torch")])
def test_dispatch_without_legacy_vendor(monkeypatch, override, expected):
    import ast
    import os
    import sys
    import types

    source = _path.with_name("npu_backend.py").read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_convolve")
    package = "_glm53_conv_dispatch_test"
    module = types.ModuleType(package + ".ascendc_conv1d")
    module.causal_conv1d_ascendc = lambda *args: "ascendc"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.delenv("GLM53_CONV_BACKEND", raising=False)
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/fla_npu/opp/vendors/fla_npu_transformer")
    if override is not None:
        monkeypatch.setenv("GLM53_CONV_BACKEND", override)
    scope = {"os": os, "torch": torch, "__package__": package, "_convolve_torch": lambda *args: "torch"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_path.with_name("npu_backend.py")), "exec"), scope)
    x = types.SimpleNamespace(device=types.SimpleNamespace(type="npu"), dtype=torch.bfloat16)
    assert scope["_convolve"](x, None, "silu", None) == expected
