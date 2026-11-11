"""Distributed forward and gradient checks for GLM5-next context parallelism."""

import os

import torch
import torch.distributed as dist


def main():
    import torch_npu
    from slime_plugins.models.glm5_next import npu_backend as backend

    rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(rank)
    dist.init_process_group("hccl")
    size = dist.get_world_size()
    torch.manual_seed(142)
    torch.set_num_threads(2)
    boundaries = torch.tensor([0, 64, 64, 256], dtype=torch.int32, device="npu")
    context = backend.build_cp_context(boundaries, dist.group.WORLD, 4)

    def leaf(value):
        return value.detach().clone().requires_grad_()

    def check(actual, expected, name, reduce=False):
        if reduce:
            dist.all_reduce(actual)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.04, atol=0.006, msg=name)

    x = (torch.randn(1, 256, 128) * 0.1).bfloat16().npu()
    weight = (torch.randn(128, 4) * 0.1).bfloat16().npu()
    dy = torch.randn_like(x)
    xr, wr = leaf(x), leaf(weight)
    reference = backend._convolve(xr, wr, "silu", boundaries)
    reference.backward(dy)
    xl, wl = leaf(x.chunk(size, 1)[rank]), leaf(weight)
    actual, _ = backend.causal_conv1d(xl, wl, "silu", context)
    actual.backward(dy.chunk(size, 1)[rank].contiguous())
    check(actual, reference.chunk(size, 1)[rank], "conv output")
    check(xl.grad, xr.grad.chunk(size, 1)[rank], "conv input gradient")
    check(wl.grad, wr.grad, "conv weight gradient", reduce=True)

    tensors = [(torch.randn(1, 256, 8, 128) * 0.1).bfloat16().npu() for _ in range(4)]
    tensors.append(torch.rand(1, 256, 8).npu())
    a = torch.zeros(8, device="npu")
    bias = torch.zeros(8 * 128, device="npu")
    grad = torch.randn_like(tensors[2])
    common = dict(
        cu_seqlens=boundaries, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0
    )
    refs = [leaf(x) for x in tensors]
    ar, br = leaf(a), leaf(bias)
    reference, _ = backend.chunk_kda(*refs[:3], g=refs[3], beta=refs[4], A_log=ar, dt_bias=br, **common)
    reference.backward(grad)
    local = [leaf(x.chunk(size, 1)[rank]) for x in tensors]
    al, bl = leaf(a), leaf(bias)
    actual, _ = backend.chunk_kda(
        *local[:3], g=local[3], beta=local[4], A_log=al, dt_bias=bl, cp_context=context, **common
    )
    actual.backward(grad.chunk(size, 1)[rank].contiguous())
    check(actual, reference.chunk(size, 1)[rank], "KDA output")
    for name, lhs, rhs in zip(["q", "k", "v", "g", "beta"], local, refs, strict=True):
        check(lhs.grad, rhs.grad.chunk(size, 1)[rank], "KDA " + name)
    check(al.grad, ar.grad, "KDA A_log", reduce=True)
    check(bl.grad, br.grad, "KDA dt_bias", reduce=True)
    torch.npu.synchronize()
    print(f"CP{size} rank {rank}: Conv1d and KDA forward/backward passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
