#!/usr/bin/env bash
# Native Slime main GLM5.3 Flash, single-node Ascend functional validation.
set -eo pipefail
SLIME_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MACHINE_CONFIG=${SLIME_MACHINE_CONFIG:-$SLIME_ROOT/deployment/machine.env}
if [[ -f "$MACHINE_CONFIG" ]]; then source "$MACHINE_CONFIG"; fi
DEPS_ROOT=${DEPS_ROOT:-$(dirname "$SLIME_ROOT")}
CONDA_ENV=${CONDA_ENV:-$DEPS_ROOT/conda_env/glm53_flash_env_cann930}
: "${CANN_ROOT:?Set CANN_ROOT in deployment/machine.env}"
: "${ATB_SET_ENV:?Set ATB_SET_ENV in deployment/machine.env}"
: "${COMM_IFNAME:?Set COMM_IFNAME in deployment/machine.env}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT in deployment/machine.env}"
: "${PROMPT_DATA:?Set PROMPT_DATA in deployment/machine.env}"
export RAY_NODE_IP=${RAY_NODE_IP:-127.0.0.1}
RAY_SHORT_DIR=${RAY_SHORT_DIR:-/tmp/slime-glm53-ray-18379}
export RAY_SHORT_DIR
mkdir -p "$DEPS_ROOT/.locks"
NPU_TASK_LOCK=${NPU_TASK_LOCK:-$DEPS_ROOT/.locks/npu-task.lock}
NPU_VALIDATION_LOCK=${NPU_VALIDATION_LOCK:-$DEPS_ROOT/.locks/npu-validation.lock}
source "$CANN_ROOT"/set_env.sh
source "$ATB_SET_ENV"
source "$CANN_ROOT"/opp/vendors/customize/bin/set_env.bash
source "$CANN_ROOT"/opp/vendors/custom_transformer/bin/set_env.bash
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$SLIME_ROOT:$DEPS_ROOT/Megatron-LM:$DEPS_ROOT/MegatronAdaptor:$DEPS_ROOT/TransformerEngineNPU:$DEPS_ROOT/Megatron-Bridge/src:$DEPS_ROOT/mbridge:$DEPS_ROOT/sglang/python:${PYTHONPATH:-}"
export SLIME_ACCELERATOR=npu
export USE_LIBUV=0
export SGLANG_MEMORY_SAVER_CUDA_GRAPH=true
# Preserve ND weights for in-place BF16 RL updates on A5.
export SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
export SGLANG_NPU_HC_PRE_TAIL_GUARD=1
export TMS_HOOK_MODE=preload
export TMS_PRESERVE_PRELOAD_MODE=1
# All serving and training ranks are local to this single-node validation.
export SLIME_HOST_IP=127.0.0.1 CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export HCCL_OP_EXPANSION_MODE=AICPU_TS HCCL_BUFFSIZE=256 DEEPEP_HCCL_BUFFSIZE=256
# Separate colocated serving and training HCCL listeners.
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60099 HCCL_NPU_SOCKET_PORT_RANGE=61000-61099
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=300 ACL_DEVICE_SYNC_TIMEOUT=300
export GLOO_SOCKET_IFNAME="$COMM_IFNAME" HCCL_SOCKET_IFNAME="$COMM_IFNAME"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0 SGLANG_ENABLE_JIT_DEEPGEMM=False
export ASCEND_USE_FIA=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy=localhost,127.0.0.1,${RAY_NODE_IP} NO_PROXY=localhost,127.0.0.1,${RAY_NODE_IP}
ulimit -n 65535
source "$SLIME_ROOT/scripts/models/glm5.3-flash-300B.sh"
for i in "${!MODEL_ARGS[@]}"; do
    if [[ ${MODEL_ARGS[$i]} == --num-experts ]]; then MODEL_ARGS[$((i+1))]=16; fi
done
RUN_DIR=${RUN_DIR:-$SLIME_ROOT/logs/$(date +%Y%m%d_%H%M%S)_glm53_main_npu}
mkdir -p "$RUN_DIR"
# Reuse the private Ray instance, or start it below when absent.
export RAY_ADDRESS="$RAY_NODE_IP:18379"
export RAY_DASHBOARD=http://127.0.0.1:19365
export RUN_DIR
cd "$SLIME_ROOT"
ARGS=(
    --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate
    --train-env-vars '{"HCCL_HOST_SOCKET_PORT_RANGE":"62000-62099","HCCL_NPU_SOCKET_PORT_RANGE":"63000-63099"}'
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT" --megatron-to-hf-mode raw
    --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label --apply-chat-template
    --rm-type "${RM_TYPE:-deepscaler}" --seed 522
    --num-rollout "${NUM_ROLLOUT:-2}" --rollout-batch-size 2 --n-samples-per-prompt 2
    --global-batch-size 4 --micro-batch-size 1
    --rollout-max-response-len 256 --rollout-temperature 1 --rollout-top-p 1
    --advantage-estimator grpo --pg-loss-type reinforce
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 12 --decoder-last-pipeline-num-layers 11
    --context-parallel-size "${CP_SIZE:-1}" --expert-model-parallel-size 2 --expert-tensor-parallel-size 1
    --allgather-cp --moe-token-dispatcher-type alltoall
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --seq-length 2048 --max-position-embeddings 2048
    --use-dynamic-batch-size --max-tokens-per-gpu 1024 --data-pad-size-multiplier 128
    --log-probs-chunk-size 128
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --no-load-optim --no-load-rng --no-save-optim --freeze-params-name-list 'model.visual.*'
    --rollout-num-gpus-per-engine 8
    --sglang-device npu --sglang-attention-backend ascend --sglang-tp-size 8
    --sglang-dp-size 1 --sglang-ep-size 1
    --sglang-mem-fraction-static 0.55 --sglang-context-length 2048
    --sglang-max-running-requests 4 --sglang-chunked-prefill-size 1024
    --sglang-disable-overlap-schedule --sglang-cuda-graph-bs 1 2 4
    --sglang-page-size 64 --sglang-kv-cache-dtype bfloat16 --sglang-watchdog-timeout 1200
    --attention-dropout 0 --hidden-dropout 0 --bf16
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --distributed-timeout-minutes 20 --update-weight-buffer-size 268435456
    --rollout-data-transport object-store
    --custom-rollout-log-function-path slime_plugins.models.glm5_next.validation.log_raw_rollouts
    --save-debug-train-data "$RUN_DIR/train-{rollout_id}.pt"
    --use-tensorboard --tb-project-name "$SLIME_ROOT/logs/tensorboard" --tb-experiment-name "$(basename "$RUN_DIR")"
    "$@"
)
printf '%q ' python "$SLIME_ROOT/train.py" "${ARGS[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"
if [[ ${DRY_RUN:-0} == 1 ]]; then cat "$RUN_DIR/command.txt"; exit 0; fi
# Both locks are shared with the existing GLM launchers on this server.
exec 9>"$NPU_TASK_LOCK"
flock -n 9 || { echo "Another task holds the NPU task lock" >&2; exit 75; }
exec 8>"$NPU_VALIDATION_LOCK"
flock -n 8 || { echo "Another task holds the validation NPU lock" >&2; exit 75; }
# Reuse a healthy private Ray; start only when no local Ray is running.
python - "$DEPS_ROOT" "$RUN_DIR" <<'PY_RAY_STATE'
import json, os, socket, sys, urllib.request
from pathlib import Path
import psutil

base, run = map(Path, sys.argv[1:])
def port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except ConnectionRefusedError:
        return False
    except OSError as exc:
        raise SystemExit(f"Cannot determine Ray endpoint {host}:{port}: {exc}")

gcs = port_open(os.environ["RAY_NODE_IP"], 18379)
dashboard = port_open("127.0.0.1", 19365)
if gcs and dashboard:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open("http://127.0.0.1:19365/api/jobs/", timeout=5) as response:
            jobs = json.load(response)
        if not isinstance(jobs, list):
            raise ValueError("Unexpected Job API response")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Existing Ray Job API is unhealthy; no restart performed: {exc}")
    active = [job.get("submission_id") for job in jobs if job.get("status") in ("RUNNING", "PENDING")]
    if active:
        raise SystemExit(f"Existing Ray has active jobs {active}; refusing duplicate launch")
    action = "reuse"
elif gcs or dashboard:
    raise SystemExit("Existing Ray is only partially available; no restart performed")
else:
    for proc in psutil.process_iter(["name", "status"]):
        try:
            if proc.info["name"] in ("raylet", "gcs_server") and proc.info["status"] != psutil.STATUS_ZOMBIE:
                raise SystemExit(f"Ray process {proc.pid} exists but private endpoints are unavailable; no restart performed")
        except psutil.NoSuchProcess:
            continue
    ray_dir = base / "ray"
    ray_dir.mkdir(exist_ok=True)
    short_dir = Path(os.environ["RAY_SHORT_DIR"])
    if short_dir.is_symlink():
        if short_dir.resolve() != ray_dir.resolve():
            raise SystemExit("Ray short path points to another directory")
    elif short_dir.exists():
        raise SystemExit("Ray short path is not our symlink")
    else:
        short_dir.symlink_to(ray_dir, target_is_directory=True)
    for port in [18379] + list(range(19365, 19373)):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                sock.listen(1)
            except OSError as exc:
                raise SystemExit(f"Ray port {port} is unavailable: {exc}")
    action = "start"
(run / "ray-action.txt").write_text(action + "\n")
print("Reusing existing private Ray at " + os.environ["RAY_NODE_IP"] + ":18379" if action == "reuse"
    else "No running Ray found; starting private Ray")
PY_RAY_STATE
if [[ $(cat "$RUN_DIR/ray-action.txt") == start ]]; then
env -u RAY_ADDRESS -u RAY_DASHBOARD_ADDRESS ray start --head \
    --node-ip-address=${RAY_NODE_IP} --port=18379 \
    --dashboard-host=127.0.0.1 --dashboard-port=19365 \
    --object-manager-port=19366 --node-manager-port=19367 \
    --ray-client-server-port=19368 \
    --dashboard-agent-grpc-port=19369 --dashboard-agent-listen-port=19370 \
    --runtime-env-agent-port=19371 --metrics-export-port=19372 \
    --min-worker-port=20000 --max-worker-port=20999 \
    --num-cpus=48 --resources='{"NPU":8}' \
    --temp-dir="$RAY_SHORT_DIR" --disable-usage-stats \
    2>&1 | tee "$RUN_DIR/ray-start.log"
python - <<'PY_RAY_READY'
import time, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
start = time.monotonic()
next_notice = 0
while time.monotonic() - start < 180:
    elapsed = int(time.monotonic() - start)
    try:
        with opener.open("http://127.0.0.1:19365/api/jobs/", timeout=2) as response:
            if response.status == 200:
                print(f"Private Ray Job API ready after {elapsed}s", flush=True)
                break
    except OSError:
        pass
    if elapsed >= next_notice:
        print(f"Waiting for private Ray Job API... {elapsed}s", flush=True)
        next_notice = elapsed + 10
    time.sleep(1)
else:
    raise SystemExit("Private Ray Job API did not become ready within 180s")
PY_RAY_READY
fi
npu-smi info > "$RUN_DIR/preflight-npu-smi.txt"
python - "$RUN_DIR" <<'PY_GATE'
import hashlib,json,re,subprocess,sys
from pathlib import Path
run=Path(sys.argv[1]);s=(run/'preflight-npu-smi.txt').read_text()
if len(re.findall('No running processes found in NPU',s)) != 8:
    raise SystemExit('NPU occupancy cannot be verified empty; waiting, no process will be killed')
hccl=Path('/etc/hccl_rootinfo.json')
record={'npu_pids':[],'locks':'both held by this launcher','hccl_sha256':hashlib.sha256(hccl.read_bytes()).hexdigest() if hccl.exists() else None}
(run/'preflight.json').write_text(json.dumps(record,indent=2))
PY_GATE
python - <<'PY' > "$RUN_DIR/runtime-env.json"
import os,json
allowed={'PATH','PYTHONPATH','SLIME_ACCELERATOR','SLIME_HOST_IP','CUDA_DEVICE_MAX_CONNECTIONS','USE_LIBUV','no_proxy','NO_PROXY','OMP_NUM_THREADS','TOKENIZERS_PARALLELISM','PYTHONDONTWRITEBYTECODE'}
prefix=('ASCEND','HCCL','GLOO','RAY_','PYTORCH_','SGLANG_','TMS_','DEEPEP_','LD_LIBRARY_PATH','LD_PRELOAD','CANN','ATB','CUSTOM','TBE','TE_')
print(json.dumps({'env_vars':{k:v for k,v in os.environ.items() if k in allowed or k.startswith(prefix)}}))
PY
python -u "$CONDA_ENV/bin/ray" job submit --address="${RAY_DASHBOARD:-http://127.0.0.1:19365}" \
    --runtime-env-json="$(cat "$RUN_DIR/runtime-env.json")" \
    -- bash "$RUN_DIR/command.txt" 2>&1 | tee "$RUN_DIR/driver.log"
