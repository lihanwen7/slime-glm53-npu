"""Numerical forward/backward checks for CANN GLM5-next mHC."""
import importlib
import sys
import types
from pathlib import Path
import pytest
import torch
pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
_root = Path(__file__).resolve().parents[1] / "slime_plugins/models/glm5_next"
_pkg = types.ModuleType("_glm53_mhc_test")
_pkg.__path__ = [str(_root)]
sys.modules.setdefault("_glm53_mhc_test", _pkg)
ref = importlib.import_module("_glm53_mhc_test.ops")
fused = importlib.import_module("_glm53_mhc_test.ascendc_mhc")


@pytest.mark.parametrize("tokens,batch", [(8, 1), (65, 2), (1024, 1)])
def test_mhc_forward_backward(tokens, batch):
    torch.manual_seed(53)
    n, d = 4, 4096
    x = torch.randn(tokens, batch, n*d, device="npu", dtype=torch.bfloat16).requires_grad_()
    w = (torch.randn(n*n+2*n, n*d, device="npu", dtype=torch.bfloat16)*0.01).requires_grad_()
    scale = torch.tensor([0.1,0.2,0.3],device="npu",dtype=torch.float32,requires_grad=True)
    bias = (torch.randn(24,device="npu",dtype=torch.float32)*0.1).requires_grad_()
    y = torch.randn(tokens,batch,d,device="npu",dtype=torch.bfloat16).requires_grad_()
    args = [x,w,scale,bias,y]
    baseline = [v.detach().clone().requires_grad_() for v in args]
    expected = ref._mhc_pre_reference(*baseline[:4],n,1e-5,1e-6,20)
    actual = fused.mhc_pre_ascendc(*args[:4],n,1e-5,1e-6,20)
    final = fused.mhc_post_ascendc(y,x,actual[1],actual[2])
    assert final._base is None, "pipeline output must be viewless"
    final_ref = ref._mhc_post_reference(baseline[4],baseline[0],expected[1],expected[2])
    grad_out = [torch.randn_like(v)*0.1 for v in (*actual,final)]
    torch.autograd.backward((*actual,final), grad_out)
    torch.autograd.backward((*expected,final_ref), grad_out)
    torch.npu.synchronize()
    pairs = list(zip(("collapsed","post","residual","merged"),(*actual,final),(*expected,final_ref)))
    pairs += [(name,a.grad,b.grad) for name,a,b in zip(("dx","dw","dscale","dbias","dy"),args,baseline)]
    for name,a,b in pairs:
        a,b=a.float(),b.float()
        rel=((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-8)).item()
        print(tokens,batch,name,"max_abs",(a-b).abs().max().item(),"relative_rms",rel,flush=True)
        assert torch.isfinite(a).all()
        assert rel < 0.001, (name,rel)
