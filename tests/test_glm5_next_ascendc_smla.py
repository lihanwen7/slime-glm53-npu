"""Optional A3 sparse-MLA numerical regression; run with GLM53_TEST_NPU=1."""

import os

import pytest


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("GLM53_TEST_NPU") != "1", reason="Set GLM53_TEST_NPU=1 on an idle A3 NPU")
def test_a3_sparse_mla_short_query_high_indices_forward_backward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an available NPU")
    device_index = torch.npu.current_device()
    if not torch.npu.get_device_name(device_index).startswith("Ascend910_93"):
        pytest.skip("requires Ascend A3 (Ascend910_93)")

    from slime_plugins.models.glm5_next.ascendc_smla import SparseMLA

    # Short Q and a large, nonmonotonic sparse selection exposed incorrect
    # native forward work partitioning, which also corrupted its gradients.
    query_tokens, kv_tokens, heads, dim = 65, 4096, 64, 512
    scale = 256**-0.5
    generator = torch.Generator(device="cpu").manual_seed(530205)
    q_cpu = torch.randn(query_tokens, heads, dim, generator=generator).to(torch.bfloat16)
    kv_cpu = torch.randn(kv_tokens, 1, dim, generator=generator).to(torch.bfloat16)
    dout_cpu = (torch.randn(query_tokens, heads, dim, generator=generator) * 0.01).to(torch.bfloat16)
    indices = torch.full((query_tokens, 1, 2112), -1, dtype=torch.int32)
    for row in range(query_tokens):
        # Queries correspond to the final 65 positions of the KV sequence.
        # Pick 512 complete pools of four keys and retain the incomplete tail:
        # each row has 2048..2051 valid slots, followed by native-width padding.
        end = kv_tokens - query_tokens + row + 1
        complete_pools = end // 4
        pools = torch.linspace(0, complete_pools - 1, 512).floor().long().roll(row % 512)
        selected = (pools[:, None] * 4 + torch.arange(4)).reshape(-1)
        selected = torch.cat((selected, torch.arange(complete_pools * 4, end)))
        indices[row, 0, : selected.numel()] = selected.to(torch.int32)

    # Independent CPU FP32 attention: one shared tensor is used as both K and V,
    # so autograd includes both contributions to its gradient. Process every
    # query/head, retaining exact BF16 inputs but bounding reference graph size.
    reference_kv = kv_cpu.float().requires_grad_()
    expected_dkv = torch.zeros_like(reference_kv)
    expected_out, expected_dq = [], []
    for row in range(query_tokens):
        query = q_cpu[row].float().requires_grad_()
        selected = indices[row, 0]
        selected = selected[selected >= 0].long()
        shared = reference_kv[selected, 0]
        output = ((query @ shared.T) * scale).softmax(-1) @ shared
        dq, dkv = torch.autograd.grad(output, (query, reference_kv), dout_cpu[row].float())
        expected_out.append(output.detach())
        expected_dq.append(dq)
        expected_dkv.add_(dkv)
    expected = (torch.stack(expected_out), torch.stack(expected_dq), expected_dkv)

    device = f"npu:{device_index}"
    with torch.npu.device(device_index):
        q = q_cpu.to(device).requires_grad_()
        kv = kv_cpu.to(device).requires_grad_()
        output, _ = SparseMLA.apply(q, kv, indices.to(device), scale)
        dq, dkv = torch.autograd.grad(output, (q, kv), dout_cpu.to(device))
        actual = tuple(tensor.detach().float().cpu() for tensor in (output, dq, dkv))

    # Observed BF16 relative RMS is below 0.003 for this fixed input. A 0.01
    # tolerance allows kernel rounding differences while rejecting the original
    # order-one forward error and several-thousand-fold gradient amplification.
    for name, result, reference in zip(("out", "dq", "dkv"), actual, expected, strict=True):
        assert result.shape == reference.shape, name
        assert torch.isfinite(result).all(), name
        error = (result - reference).double()
        relative_rms = error.norm() / reference.double().norm().clamp_min(1e-12)
        assert relative_rms < 0.01, (name, relative_rms.item())
        if name != "dkv":
            # Check each query too, so a few bad work-partition boundaries cannot
            # be hidden by averaging over correctly computed queries.
            query_error = error.flatten(1).norm(dim=1)
            query_norm = reference.double().flatten(1).norm(dim=1).clamp_min(1e-12)
            worst_query_rms = (query_error / query_norm).max()
            assert worst_query_rms < 0.01, (name, worst_query_rms.item())
