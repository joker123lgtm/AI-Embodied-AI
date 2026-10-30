# RoboTwin + ACT Baseline：调试与评测命令记录

> 环境：AutoDL Linux，`/root/autodl-tmp/AI/RoboTwin`，Conda 环境 `RoboTwin`，GPU Tesla V100S 32GB。
>
> 项目：`beat_block_hammer`；策略 ACT；训练权重位于 `demo_clean-beat_block_hammer-aloha_agilex-joint-0`；动作空间 `joint`；`chunk_size=50`；`temporal_agg=true`。
>
> **当前状态**：ACT 策略服务与模型权重加载曾成功；Conda 子进程初始化和 cuRobo 导入问题已排查并修复/验证。5 个 Episode 的正式任务成功率**尚未取得**，不要将调度器的 `success=1` 当作任务成功率。

## 1. 登录后激活环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
cd /root/autodl-tmp/AI/RoboTwin
```

## 2. 检查 GPU、Checkpoint 与部署配置

```bash
nvidia-smi
find XPolicyLab/policy/ACT/checkpoints -name '*.ckpt' -printf '%p  %s bytes\n'
cat XPolicyLab/policy/ACT/deploy.yml
```

主要 Checkpoint：

```text
XPolicyLab/policy/ACT/checkpoints/demo_clean-beat_block_hammer-aloha_agilex-joint-0/policy_epoch_6000_seed_0.ckpt
```

## 3. 仅在出现 Conda 子进程初始化错误时：修复启动脚本

典型错误：`conda: command not found` 或 `/etc/profile.d/conda.sh: No such file or directory`。

确认脚本内容和路径：

```bash
nl -ba XPolicyLab/utils/run_sim_env_client.sh | sed -n '1,45p'
ls -l /root/miniconda3/etc/profile.d/conda.sh
```

若第 18 行仍是 `source "$(conda info --base)/etc/profile.d/conda.sh"`，可以先备份，再修改：

```bash
cp -n XPolicyLab/utils/run_sim_env_client.sh XPolicyLab/utils/run_sim_env_client.sh.bak
sed -i '18c\source /root/miniconda3/etc/profile.d/conda.sh' XPolicyLab/utils/run_sim_env_client.sh
bash -n XPolicyLab/utils/run_sim_env_client.sh
echo $?
```

预期语法检查退出码为 `0`。**不要在已正确修复的脚本上反复修改**；如目录与行号改变，先重新查看内容。

## 4. 修复 cuRobo 的 Python 导入路径

遇到的错误：

```text
ImportError: cannot import name 'CuroboPlanner'
ModuleNotFoundError: No module named 'curobo.types.math'; 'curobo.types' is not a package
```

原因：服务器上存在多套 cuRobo 源码；正确的 v0.7.8 代码位于 `envs/curobo/src/curobo/`，但 Python 之前错误加载了 `envs/curobo/curobo/curobo/types.py`。

在 **RoboTwin 项目根目录** 执行：

```bash
export PYTHONPATH="$PWD/envs/curobo/src:${PYTHONPATH:-}"
```

验证：

```bash
python - <<'PY'
import curobo
from curobo.types.math import Pose
from envs.robot.planner import CuroboPlanner
print('cuRobo path:', curobo.__file__)
print('CuroboPlanner imported successfully!')
PY
```

检查源码版本和目录（只读）：

```bash
git -C envs/curobo describe --tags --always
find envs/curobo -maxdepth 5 \( -name math.py -o -name types.py \) -print | head -40
```

**注意**：`export PYTHONPATH` 不是永久配置，新终端通常需要重新运行；不要随意删除另一套源码或覆盖 PyTorch/CUDA 依赖。

## 5. 创建单任务评测配置（只需一次）

不使用包含多个 GPU 和大量任务的 `env_cfg/eval/all_tasks.yml`；新建独立文件：

```bash
cat > env_cfg/eval/act_baseline.yml <<'EOF_CONFIG'
gpu_ids: "0"
jobs_per_gpu: 1
num_workers: 1
enable_remote: false
policy_server_ip: 127.0.0.1
policy_server_port: [18080]

tasks:
  - beat_block_hammer
EOF_CONFIG

cat env_cfg/eval/act_baseline.yml
```

## 6. Dry Run：只检查调度，不启动模型与仿真

```bash
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/act_baseline.yml \
  --policy-name ACT \
  --ckpt-name demo_clean-beat_block_hammer-aloha_agilex-joint-0 \
  --env-cfg-type arx_x5 \
  --action-type joint \
  --policy-conda-env RoboTwin \
  --eval-env-conda-env RoboTwin \
  --task-config demo_clean \
  --test-num 5 \
  --seed 0 \
  --dry-run
```

之前观察到调度器输出 `Jobs: 1`、`GPUs: {'0': 1}`、`test_num: 5`、`eval_batch: False`、`num_workers: 1`、`expert_check: True`。

**说明**：`--env-cfg-type arx_x5` 是该项目策略接口所用的动作配置标识；你的日志显示仿真实际 Embodiment 为 `aloha-agilex`；`--action-type joint` 明确表示关节空间动作。Dry Run 仅证明调度参数可解析，不证明仿真能成功启动。

## 7. 正式运行 5 Episode ACT Baseline

确认第 3、4 步修复有效，并在同一终端设置 PYTHONPATH 后执行：

```bash
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/act_baseline.yml \
  --policy-name ACT \
  --ckpt-name demo_clean-beat_block_hammer-aloha_agilex-joint-0 \
  --env-cfg-type arx_x5 \
  --action-type joint \
  --policy-conda-env RoboTwin \
  --eval-env-conda-env RoboTwin \
  --task-config demo_clean \
  --test-num 5 \
  --seed 0 \
  --stream-output
```

重点查看策略服务就绪、仿真客户端连接、每次 Episode 的 `success` 和最终 `Success rate` / `Final success rate`。如报错，优先定位第一条完整异常。

## 8. 查找评测结果

```bash
ls -lt eval_result/multitask/
find eval_result -name '_result.txt'
find eval_result -name '_result.txt' -exec sh -c 'echo "$1"; cat "$1"; echo' _ {} \;
```

在批量评测路径中，`_result.txt` 保存 `success_num/completed`，例如 `0.6` 表示 60%；但**不同评测路径可能保存位置不同**。没有结果文件时，应先检查日志或作业 summary，而不是推定模型失败。

任务成功率：

```text
Success Rate = 成功 Episode 数 / 已完成 Episode 数 × 100%
```

`beat_block_hammer` 的 `check_success()` 判定包含：锤子功能点与方块功能点在 X、Y 方向的偏差分别小于 0.02 m，且锤子与方块发生接触。

## 9. 以后重新登录服务器的最简流程

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
cd /root/autodl-tmp/AI/RoboTwin
export PYTHONPATH="$PWD/envs/curobo/src:${PYTHONPATH:-}"
```

随后直接执行 **第 7 步** 正式评测命令即可（前提：单任务配置已存在、Conda 脚本已修复）。

## 10. 后续优化实验建议

先取得 Baseline 的完整成功率；然后固定任务配置、Checkpoint、随机种子与 Episode 数量，仅改变 `temporal_agg`，分别测量 `true` 与 `false`。5 个 Episode 适合流程验证，不能单凭少量样本断言哪个配置性能更优。
