"""Tests for model backend selection and request validation."""

import ast
import builtins
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_definitions(relative_path, names, namespace):
    tree = ast.parse((ROOT / relative_path).read_text())
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(tree, relative_path, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("backend", ["npu", "cuda", "cpu"])
@pytest.mark.parametrize("setting", [None, "auto", "npu", "cuda"])
def test_attention_uses_framework_backend(monkeypatch, backend, setting):
    if setting is None:
        monkeypatch.delenv("SLIME_ACCELERATOR", raising=False)
    else:
        monkeypatch.setenv("SLIME_ACCELERATOR", setting)
    path = ROOT / "slime_plugins/models/glm5_next/attention.py"
    node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.If))
    imported = []

    def import_backend(name, globals=None, locals=None, fromlist=(), level=0):
        imported.append(name)
        return SimpleNamespace(**{name: object() for name in fromlist})

    ns = dict(
        accelerator=SimpleNamespace(device_type=lambda: backend),
        __builtins__={**vars(builtins), "__import__": import_backend},
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    assert ("npu_backend" in imported) == (backend == "npu")
    assert ("fla.modules" in imported) == (backend != "npu")


@pytest.mark.parametrize("backend,expected", [("npu", 1e-5), ("cuda", 1e-6), ("cpu", 1e-6)])
def test_vision_normalization_uses_framework_backend(backend, expected):
    path = ROOT / "slime_plugins/models/glm5_next/vision.py"
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "qk_norm_eps" for t in node.targets)
    )
    ns = dict(accelerator=SimpleNamespace(device_type=lambda: backend), config=SimpleNamespace(rms_norm_eps=1e-5))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    assert ns["qk_norm_eps"] == expected


@pytest.mark.parametrize("boundaries", [[], [0], [0, 8, 4, 12], [1, 12], [0, 13], [0, 1.5, 12]])
def test_invalid_packed_boundaries(boundaries):
    ns = load_definitions("slime_plugins/models/glm5_next/npu_backend.py", {"_boundaries"}, {})
    with pytest.raises(ValueError):
        ns["_boundaries"](boundaries, 12)


def test_empty_packed_sequences_preserve_token_offsets():
    ns = load_definitions("slime_plugins/models/glm5_next/npu_backend.py", {"_boundaries"}, {})
    assert ns["_boundaries"]([0, 0, 4, 4, 12], 12) == [0, 0, 4, 4, 12]
    assert ns["_boundaries"](None, 12) == [0, 12]


def test_convolution_without_cp_context():
    calls = []
    ns = load_definitions(
        "slime_plugins/models/glm5_next/npu_backend.py",
        {"causal_conv1d"},
        dict(
            _can_shard_heads=lambda *a: False,
            _gather=lambda x, c: x,
            _local=lambda x, c: x,
            _convolve=lambda *args: calls.append(args) or "output",
        ),
    )
    assert ns["causal_conv1d"]("input", "weight", "silu", None) == ("output", None)
    assert calls == [("input", "weight", "silu", None)]








@pytest.mark.parametrize("device", ["cpu", "cuda", "npu"])
def test_npu_mhc_setting_does_not_affect_other_devices(monkeypatch, device):
    monkeypatch.setenv("GLM53_MHC_BACKEND", "invalid")
    ns = load_definitions("slime_plugins/models/glm5_next/ops.py", {"_use_fused_mhc"}, dict(os=os))
    tensor = SimpleNamespace(device=SimpleNamespace(type=device))
    if device == "npu":
        with pytest.raises(ValueError, match="GLM53_MHC_BACKEND"):
            ns["_use_fused_mhc"](tensor)
    else:
        assert ns["_use_fused_mhc"](tensor) is False


def test_device_probe_does_not_swallow_interrupt():
    def interrupted():
        raise KeyboardInterrupt

    import functools

    ns = load_definitions(
        "slime_plugins/models/glm5_next/ascendc_fla_utils.py",
        {"get_available_device"},
        dict(
            functools=functools,
            triton=SimpleNamespace(
                runtime=SimpleNamespace(driver=SimpleNamespace(active=SimpleNamespace(get_current_target=interrupted)))
            ),
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        ns["get_available_device"]()


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu"])
def test_npu_kpool_setting_does_not_affect_other_devices(monkeypatch, device):
    monkeypatch.setenv("GLM53_KPOOL_BACKEND", "invalid")
    tree = ast.parse((ROOT / "slime_plugins/models/glm5_next/attention.py").read_text())
    cls = next(node for node in tree.body if getattr(node, "name", None) == "PooledIndexer")
    forward = next(node for node in cls.body if getattr(node, "name", None) == "forward")
    start = next(
        i
        for i, node in enumerate(forward.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "backend"
    )
    code = compile(ast.Module(body=forward.body[start : start + 2], type_ignores=[]), "kpool_selection", "exec")
    ns = dict(os=os, x=SimpleNamespace(device=SimpleNamespace(type=device)))
    if device == "npu":
        with pytest.raises(ValueError, match="GLM53_KPOOL_BACKEND"):
            exec(code, ns)
    else:
        exec(code, ns)
        assert ns["backend"] == "torch"
