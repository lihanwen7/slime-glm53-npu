# GLM5.3 Flash 单机 NPU 迁移说明

本私仓保留 Slime 原有提交历史及已移植的 6 个 Ascend 补丁提交，并提交当前工作区中的 GLM5-next NPU 代码。其他仓库固定 commit，并通过 `patches/sources.lock.json` 中的补丁顺序恢复。不要重新运行旧的 slime-ascend `build.sh`，也不要再次向本私仓打那 6 个 Slime 补丁。

## 1. 目录布局

```text
WORKSPACE/
  slime/                   本私仓
  sglang/
  sgl-kernel-npu-src/       仅从此固定源码重建 TMS
  Megatron-LM/
  MegatronAdaptor/
  TransformerEngineNPU/
  Megatron-Bridge/
  mbridge/
  conda_env/glm53_flash_env_cann930/
```

模型、训练数据、CANN、ATB 和 wheel 制品不放进 Git。`environment/runtime-versions.json` 是实际安装版本记录，`environment/native-artifacts.json` 是服务器上找到的相关 wheel 校验记录；它们不是可以直接执行的全量安装锁文件，也不保证公开包源提供相同 vendor 二进制。

## 2. 恢复其他仓库并打补丁

在克隆出的本私仓根目录执行，先用实际目录替换 `/path/to/WORKSPACE`：

```bash
# 克隆缺失仓库，检出固定 commit，然后在临时副本中检查补丁顺序。
python deployment/apply_dependency_patches.py \
  --workspace /path/to/WORKSPACE --clone

# 正式应用补丁。只接受干净、HEAD 精确匹配的依赖工作区。
python deployment/apply_dependency_patches.py \
  --workspace /path/to/WORKSPACE --apply
```

脚本检查每份补丁的 SHA256、基线 commit 和补丁后文件 SHA256。不会 reset、清理或覆盖已有脏工作区；已完整匹配快照的仓库不重复打补丁。默认不指定 `--apply` 时，在临时副本重放补丁。单独处理一个依赖可加 `--dependency sglang` 或 `--dependency sgl-kernel-npu`。

手动应用时，在对应依赖仓库根目录，先检出 `sources.lock.json` 的 commit，再按 `patches` 数组顺序对每份补丁执行：

```bash
git apply --check /absolute/path/to/slime/patches/third_party/DEPENDENCY/PATCH.patch
git apply /absolute/path/to/slime/patches/third_party/DEPENDENCY/PATCH.patch
```

不要同时应用 `patches/reference` 下的原始补丁；该目录仅用于说明来源。不要把其他版本的旧 Megatron、Qwen/VL 或 checkpoint 补丁追加到本清单。

## 3. 环境和 TMS

目标机器先准备匹配的 Linux/aarch64、NPU/驱动、CANN 9.3.0、ATB 和 customize/custom_transformer 自定义算子。不要从这台服务器复制 `/etc/hccl_rootinfo.json` 覆盖另一台机器的系统配置。

可用目标机器验证过的 vendor wheel 重建 Conda 环境，也可另行用 `conda-pack` 搬迁原环境后执行 `conda-unpack`；Conda 包之外的源码路径、CANN/驱动及自定义算子仍必须重新配置。原环境中的 editable 安装可能保留服务器绝对路径，需要重新安装或由启动脚本的源码路径覆盖。不要只凭版本号认定 wheel 与目标 SoC/ABI 一致。

TMS 必须来自本次固定源码和三份补丁。源码已应用补丁后，在目标环境加载 CANN，再单独构建安装 TMS：

```bash
source /path/to/cann-9.3.0/set_env.sh
cd /path/to/WORKSPACE/sgl-kernel-npu-src/contrib/torch_memory_saver/python
python -m pip wheel --no-deps --no-build-isolation . -w /path/to/tms-wheels
# 选择刚刚为目标机器构建出的那个 wheel，明确写出文件名后安装。
python -m pip install --no-deps /path/to/tms-wheels/torch_memory_saver-0.0.8-PLATFORM.whl

cd /path/to/WORKSPACE/slime
python -m pip install --no-deps -e .
```

上面的 `PLATFORM` 是说明用占位符，必须替换为实际 wheel 文件名。不要运行 sgl-kernel-npu 的整包构建来代替 TMS 单独构建；其他内核使用与现环境对应、经过校验的制品。FLA-NPU、Triton Ascend 和其他扩展的安装应按目标硬件验证；本导出不自动重装或编译这些包。

## 4. 配置机器并启动

```bash
cp deployment/machine.env.example deployment/machine.env
# 编辑 machine.env 中的环境根目录、CANN/ATB、网卡、模型和数据位置。
```

`machine.env` 是持久机器配置文件，已加入 `.gitignore`。不要提交账号凭据。共享机器上，`NPU_TASK_LOCK` 和 `NPU_VALIDATION_LOCK` 应指向其他任务共同遵守的锁文件；默认锁仅约束本 WORKSPACE 的启动器。

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda activate /path/to/WORKSPACE/conda_env/glm53_flash_env_cann930

# 仅生成最终命令，检查配置，不启动 Ray 或占用 NPU。
DRY_RUN=1 bash scripts/run-glm53-main-npu.sh \
  --rollout-session-id glm53_check_001

# 正式启动，每次新任务使用唯一的 session ID。
bash scripts/run-glm53-main-npu.sh \
  --rollout-session-id "glm53_manual_$(date +%Y%m%d_%H%M%S)"
```

启动器加载 CANN/ATB、自定义算子和源码路径，获取双锁，复用健康的私有 Ray；没有 Ray 才启动。已有活动 Ray job、部分存活的异常 Ray 或 NPU 占用时退出，不自动终止其他任务。新建 Ray 的 dashboard 可能启动较慢，等待期间每 10 秒输出进度。

当前配置为 2 轮、回答上限 256 token、上下文/训练长度 2048、BF16、训练 TP1/PP4/CP1/EP2、推理 TP8/DP1/EP1、每轮 2 题各 2 回答、global batch 4/micro batch 1。图和物理卸载开启。修改固定参数应提交配置变更，便于复现。

`RUN_DIR`、`NUM_ROLLOUT`、`CP_SIZE`、`RM_TYPE` 仍可由机器配置或调用环境覆盖；回答长度不读取 `RESPONSE_LENGTH`。启动前核对生成的 `command.txt` 与 `runtime-env.json`。长跑可明确设置 `NUM_ROLLOUT=10000`，但应先完成目标机器两步验证。

## 5. 不要遗漏的开关

- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False`：此前对照验证影响 TMS 实际物理卸载。
- `TMS_HOOK_MODE=preload`、`TMS_PRESERVE_PRELOAD_MODE=1`：防止导入时覆盖卸载模式。
- `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1`：使用已有 ND 权重路径，支持分片原地更新。
- `SGLANG_NPU_HC_PRE_TAIL_GUARD=1`：Python 侧增加 HcPreSinkhorn 输入 backing storage，保持逻辑 shape。
- serving/train HCCL 使用 60000/61000 与 62000/63000 两组独立端口范围。

`HCCL_ALGO`、`SGLANG_NPU_GRAPH_CPU_BACKUP`、`SGLANG_NPU_PRE_OFFLOAD_FENCE` 是历史排障变量，正式启动器不主动开启。机器配置示例显式 unset 它们，避免终端残留。CPU 备份和卸载前 fence 在完整模型对照中未解决 HCCL 故障。

## 6. 已知验证边界

这是一份功能适配快照，不是已完成精度或长期稳定性验收的发布版本：

- 2026-10-19 短配置再次完成两步；原始 `grad_norm` 和 `pg_loss` 为 0，logprob 差约 0.21644/0.50547。
- 历史最长完成 24 步后，decode 出现 HCCL OneShot SendRecv/hcclRet4。根因未定。
- 回答 8192、上下文 16384 时，首次训练 L2Norm 出现 `coreDim=67072 > 65535`。此问题没有修复，源码保留原有算法和网格。
- 当前 graph CPU backup/pre-offload fence 代码仅为 opt-in 诊断功能，不代表故障修复已验证。
- 目标机器必须重新验证图、卸载、真实 rollout/logprob、训练前后向、optimizer step、权重同步和下一轮生成。

原服务器启动脚本仅存档在 `deployment/reference`，包含原机器绝对路径，不作为迁移入口。仓库未收录模型、数据集、运行日志、Conda 环境、wheel 或账号凭据。

适配代码逐项来源见 [PROVENANCE.md](PROVENANCE.md)，本次导出检查结果见 [export-validation.json](export-validation.json)。
