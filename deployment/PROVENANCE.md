# 适配代码来源

这份快照区分已有代码、已有 Ascend 补丁移植和本次新增适配，不把所有差异都归为新写代码。具体基线、补丁顺序及 SHA256 以 `patches/sources.lock.json` 和 `deployment/source-provenance.json` 为准。

| 范围 | 来源与处理 |
| --- | --- |
| Slime 主干 | 基线 `b2182cc5db977553455eb6e2e5c381eefbb35cfc`，完整 Git 历史保留。 |
| 已有 6 个 Slime Ascend 补丁 | 已移植为基线后的 6 个提交，截至 `ed5f0a96199048df3ff14a41764a9c6a1a0aa4e2`；原补丁另存 `patches/reference/slime-ascend/`，仅供追溯，不重复应用。 |
| Slime NPU 模型接线 | `glm5_next/npu_backend.py`、`attention.py`、`ops.py` 中的 GLM5-next NPU 路径及相关运行时调整作为新增适配提交。并非所有底层算子都是新写。 |
| AscendC KDA 与辅助代码 | 复用已有 KDA 桥接及已安装内核包中的 helper。`ascendc_fla_utils.py` 为原文件复制；`ascendc_l2norm.py` 只隔离 helper 导入；`ascendc_kda.py` 采用本地 helper 导入以避开无关 autotune 初始化。保留原版权，不宣称重写内核算法或修复长序列网格上限。 |
| Slime 权重同步及 rollout | NPU 私有池内 empty_cache 延后至池外执行；straw 专属数据源延迟导入；原始 rollout 指标记录等兼容修复。 |
| SGLang | 前两份补丁分别为已有 TMS 参数适配和 offload 幂等移植；第三份记录本次 HcPreSinkhorn 尾部 backing storage 保护及 opt-in 排障代码。图 CPU 备份和卸载前 fence 未解决完整模型 HCCL 故障。ND 权重使用库中已有环境开关。 |
| Torch Memory Saver | 基线仓库 `sgl-kernel-npu` 的前两份补丁为已有生命周期和图接口/preload 适配；第三份新增显式 preload 模式保护。只构建其 TMS 子项目。 |
| Megatron-LM / MegatronAdaptor / TransformerEngineNPU / Megatron-Bridge / mbridge | 当前导出中均为固定 commit、无工作区补丁。曾尝试后撤回的改动不算生效适配。 |
| CANN / ATB / custom_transformer | 使用已有部署和自定义算子，没有提交 CANN 源码修改或重新编译其算子。 |
| 启动与迁移工具 | 当前服务器启动器留档；迁移入口将机器路径改为持久配置文件。依赖补丁检查/应用工具和迁移说明为本次新增。 |

`1fe9a370c15ee3234851bc251d12507a97ac101f` 是当前运行代码改动的归档提交，不能仅凭该提交作者推断其中复制代码的原始作者。迁移包装另作提交。此快照通过短配置执行验证，精度、有效学习及长期稳定性仍有已知限制，详见 `deployment/README.md`。
