# Dependency patches

Apply only the ordered patch lists in `sources.lock.json`, using `deployment/apply_dependency_patches.py`. `reference/` contains source provenance, not additional install steps.

| Dependency | Exact base commit | Patch order |
|---|---|---|
| sglang | `872f1721cc6d4679f008167386ec6699a6580102` | `0001-tms-adapter-arguments.patch` → `0002-offload-idempotence-port.patch` → `0003-glm53-local-fixes-and-opt-in-diagnostics.patch` |
| sgl-kernel-npu | `9765e27532b8f854717eba1d3fb9a0bb97a0c887` | `0001-fix-torch-memory-saver-allocation-lifecycle.patch` → `0002-fix-tms-sglang-cuda-graph-api-and-preload-mode.patch` → `0003-preserve-explicit-preload-mode.patch` |
| Megatron-LM | `4b4acac9a1d28ea6829c8d4f566d75698a21249d` | No patches; pinned source only |
| MegatronAdaptor | `f710cd34ed54eca6b7b7ae1b51a19840d15aedbc` | No patches; pinned source only |
| TransformerEngineNPU | `fd09a928c1e2c74fa13f08dd96a9d7c68cabb48f` | No patches; pinned source only |
| Megatron-Bridge | `35b4ebfc486fb15dcc0273ceea804c3606be948a` | No patches; pinned source only |
| mbridge | `89eb10887887bc74853f89a4de258c0702932a1c` | No patches; pinned source only |

For full commands, see [migration instructions](../deployment/README.md#2-恢复其他仓库并打补丁).

SGLang 0001 and TMS 0001/0002 are existing Ascend patches. SGLang 0002 is a port of existing idempotence logic. SGLang 0003 and TMS 0003 contain this session’s additional fixes/diagnostics. The six Slime Ascend ports are already in this repository’s commit ancestry; do not apply them again.
