"""NPU numerical and packed/CP contract checks for CANN KPool."""
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
root = Path(__file__).resolve().parents[1] / "slime_plugins/models/glm5_next"
pkg = types.ModuleType("_glm53_kpool_test")
pkg.__path__ = [str(root)]
sys.modules.setdefault(pkg.__name__, pkg)
fused = importlib.import_module(pkg.__name__ + ".ascendc_kpool")
ref = importlib.import_module(pkg.__name__ + ".ops")


def inputs(length=2305):
    torch.manual_seed(53)
    device = "npu"
    x = torch.randn(length, 4096, device=device, dtype=torch.bfloat16)
    wk = torch.randn(128, 4096, device=device, dtype=torch.bfloat16) * 0.01
    gate = torch.randn_like(wk) * 0.02
    ape = torch.randn(4, 128, device=device) * 0.1
    nw = torch.ones(128, device=device, dtype=torch.bfloat16)
    nb = torch.zeros_like(nw)
    k = F.layer_norm(F.linear(x, wk), (128,), nw, nb, 1e-6)
    gates = F.linear(x, gate)
    q = torch.randn(length, 32, 128, device=device, dtype=torch.bfloat16)
    weights = torch.randn(length, 32, device=device) / 64
    return x, wk, gate, ape, nw, nb, k, gates, q, weights


@pytest.mark.parametrize("length", [3, 4, 17, 260, 2305])
def test_key_pool(length):
    x, wk, gate, ape, nw, nb, k, gates, _, _ = inputs(length)
    actual = fused.key_pool_document(x, wk, gate, ape, nw, nb, 1e-6, 4)
    count = length // 4
    assert actual.shape == (count, 128)
    if count:
        expected = (k[:count*4].float().reshape(count, 4, 128) *
                    (gates[:count*4].float().reshape(count, 4, 128) + ape).softmax(1)).sum(1).to(k.dtype)
        rel = ((actual.float()-expected.float()).square().mean().sqrt() /
               expected.float().square().mean().sqrt()).item()
        print("KEY_POOL", length, "relative_rms", rel,
              "max_abs", (actual.float()-expected.float()).abs().max().item(), flush=True)
        assert torch.isfinite(actual).all()
        assert rel < 0.01


@pytest.mark.parametrize("query_start,query_end", [(0, 2572), (2, 6), (127, 1600), (1600, 2572)])
def test_packed_indexer(query_start, query_end):
    x, wk, gate, ape, nw, nb, k, gates, q, weights = inputs(2572)
    cu = torch.tensor([0, 7, 267, 267, 2572], dtype=torch.int32, device=x.device)
    q = q[query_start:query_end]
    weights = weights[query_start:query_end]
    expected = ref.pooled_indexer_topk(q, k, weights, gates, ape, cu, 2048, 4,
                                     query_start=query_start)
    actual = fused.fused_kpool_indexer(q, x, weights, wk, gate, ape, nw, nb,
                                      1e-6, cu, 2048, 4, query_start=query_start)
    a, b = actual.cpu(), expected.cpu()
    assert a.shape == b.shape
    total = matched = 0
    for row in range(a.shape[0]):
        position = query_start + row
        start = 0 if position < 7 else (7 if position < 267 else 267)
        selected = a[row, 0][a[row, 0] >= 0].tolist()
        assert len(selected) == len(set(selected)), ("duplicate", row)
        assert all(start <= v <= position for v in selected), ("causal/document", row)
        sa, sb = set(selected), set(b[row, 0][b[row, 0] >= 0].tolist())
        assert len(sa) == len(sb), ("missing keys", row, len(sa), len(sb))
        if position-start+1 <= 2048:
            assert sa == sb, ("all visible keys must be selected", row)
        total += len(sb)
        matched += len(sa & sb)
    overlap = matched / max(total, 1)
    print("INDEXER", query_start, query_end, "set_overlap", overlap, flush=True)
    assert overlap > 0.99
