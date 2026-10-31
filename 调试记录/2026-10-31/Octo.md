# Octo 部署与 Demo 学习笔记

> 项目：具身模型论文阅读 ｜ 模型：[Octo: An Open-Source Generalist Robot Policy](https://github.com/octo-models/octo)  
> 记录日期：2026-10-31 ｜ 环境：AutoDL / Ubuntu 22.04 / Tesla V100S 32GB

## 1. 实验结果概览

已完成：

- 在 Conda 独立环境中部署 Octo；
- 配置 JAX CUDA 后端，在 V100S 上运行矩阵乘法；
- 通过 Hugging Face 镜像加载 **Octo-Small 1.5** 预训练模型；
- 输入一张 RGB 图像和文本指令 `pick up the fork`，生成 **4 步 × 7 维**的 Action Chunk；
- 保存动作数组并绘制末端位移、旋转曲线。

**注意**：目前验证的是离线动作生成，不是机器人或仿真环境中的任务成功率。

## 2. 实验环境和关键依赖

| 项目 | 实际配置 / 版本 |
|---|---|
| 服务器 | AutoDL |
| GPU | Tesla V100S-PCIE-32GB |
| OS | Ubuntu 22.04.4 LTS |
| NVIDIA 驱动 | 580.105.08 |
| 系统 CUDA Toolkit | 12.4（`nvcc --version`） |
| Conda 环境 | `octo` |
| Python | 3.10.22 |
| JAX | 0.4.20 |
| jaxlib | 0.4.20+cuda12.cudnn89 |
| nvidia-cudnn-cu12 | 8.9.7.29（Conda 环境中的 Python 包） |
| NumPy | 1.24.3 |
| ml-dtypes | 0.2.0 |
| SciPy | 1.11.4 |
| TensorFlow | 2.15.0 |
| tensorflow-metadata | 1.15.0 |
| protobuf | 3.20.3 |
| Transformers | 4.35.2 |
| WandB | 0.16.6 |

系统 `nvidia-smi` 显示的 CUDA 13.0 是驱动支持的 CUDA 版本，**不是**实际 `nvcc` 工具链版本。上述版本是此次成功实验的记录，并不是要求对任意新服务器无条件重复安装。

## 3. 首次部署 Octo

### 3.1 创建环境

```bash
conda create -n octo python=3.10 -y
conda activate octo
```

### 3.2 获取源码

```bash
mkdir -p /root/autodl-tmp/AI
cd /root/autodl-tmp/AI
git clone https://github.com/octo-models/octo.git
cd octo
```

### 3.3 安装项目和 Python 依赖

由于服务器的默认清华 PyPI 镜像曾出现无法找到 `setuptools`、`gym` 等包的情况，本次使用官方 PyPI。`dlimp` 是从 GitHub 固定提交安装的依赖，GitHub 网络超时可能导致安装失败，重试后已成功。

```bash
cd /root/autodl-tmp/AI/octo
python -m pip install -e . --no-deps --no-build-isolation

python -m pip install -r requirements.txt \
  --no-build-isolation \
  -i https://pypi.org/simple
```

**提醒**：这些是首次部署命令；不要在已成功部署的环境里反复运行安装命令，以免将已固定好的依赖重新升级。

## 4. JAX GPU 配置和验证

### 4.1 本次成功使用的 GPU 构建

Octo 源码依赖 JAX 0.4.20；曾遇到 `jax=0.4.20` 与 `jaxlib=0.4.38` 不兼容的问题。最后使用的 GPU 构建为 `jaxlib=0.4.20+cuda12.cudnn89`。

此前采用的安装命令（**故障排查记录；不是日常启动命令**）：

```bash
python -m pip install --force-reinstall \
  'jax[cuda12_pip]==0.4.20' \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  -i https://pypi.org/simple
```

安装过程中曾意外把 NumPy、ml-dtypes 升级至与 TensorFlow 2.15.0 冲突的版本，随后恢复：

```bash
python -m pip install \
  'numpy==1.24.3' 'ml-dtypes==0.2.0' \
  --no-deps -i https://pypi.org/simple
```

### 4.2 cuDNN 不匹配的修复

曾出现：

```text
Found cuDNN version 0, but JAX was built against version 8904
```

当时环境中只有 `libcudnn.so.9`，而此次 JAX GPU 构建需要 cuDNN 8.9。采用：

```bash
python -m pip install 'nvidia-cudnn-cu12==8.9.7.29' \
  --force-reinstall --no-deps -i https://pypi.org/simple
```

为了让运行时找到 Conda 环境中的 `libcudnn.so.8`，设置：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cuda
```

### 4.3 GPU 真正计算验证

```bash
python - <<'PY'
import jax
import jax.numpy as jnp

print('JAX:', jax.__version__)
print('Devices:', jax.devices())
print('Backend:', jax.default_backend())
a = jnp.ones((1000, 1000))
b = jnp.ones((1000, 1000))
c = jax.jit(lambda x, y: x @ y)(a, b)
c.block_until_ready()
print('Result shape:', c.shape)
print('Result device:', c.device())
PY
```

本次成功输出：

```text
JAX: 0.4.20
Devices: [cuda(id=0)]
Backend: gpu
Result shape: (1000, 1000)
Result device: cuda:0
```

**只有 `CpuDevice(id=0)` 时，程序仍在 CPU 上，不能凭一行手写 `print('GPU computation successful!')` 判断 GPU 成功。**

## 5. 其他兼容性故障与修复记录

这些命令是本次排障中使用过的版本固定方法；如果现有环境正常，**不要再次批量执行**。

| 报错 / 现象 | 根因或定位 | 处理 |
|---|---|---|
| `scipy.linalg` 没有 `tril` | 新版 SciPy 移除了旧接口 | 固定 SciPy 1.11.4 |
| `cannot import FlaxAutoModel` | Transformers 新版不再提供所需的旧 Flax 接口 | 固定 Transformers 4.35.2 |
| `cannot import runtime_version from google.protobuf` | TensorFlow Metadata / protobuf 不兼容 | 固定相关兼容版本 |
| `wandb` 与 protobuf 版本冲突 | WandB 新版要求较高的 protobuf | 固定 WandB 0.16.6，并补齐其依赖 |
| `ModuleNotFoundError: appdirs` | 曾使用 `--no-deps` 安装 WandB | 重新允许 pip 安装其依赖 |
| TensorRT 未检测到 | TensorFlow 推理优化组件未安装 | 当前 JAX 推理不依赖它，可忽略警告 |

本次使用过的修复命令：

```bash
python -m pip install 'scipy==1.11.4' --no-deps -i https://pypi.org/simple
python -m pip install 'transformers==4.35.2' -i https://pypi.org/simple
python -m pip install 'tensorflow-metadata==1.15.0' --no-deps -i https://pypi.org/simple
python -m pip install 'protobuf==3.20.3' --no-deps -i https://pypi.org/simple
python -m pip install 'wandb==0.16.6' 'protobuf==3.20.3' -i https://pypi.org/simple
```

验证：

```bash
python -m pip check
python - <<'PY'
from octo.model.octo_model import OctoModel
print('OctoModel imported successfully!')
PY
```

## 6. 网络：使用 Hugging Face 镜像

服务器曾无法连接 `huggingface.co:443`，但第三方镜像 `hf-mirror.com` 可以访问。加载公开模型时设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**注意**：`hf-mirror.com` 为第三方镜像，非 Hugging Face 官方服务；模型文件下载仍取决于实际网络和镜像可用性。

## 7. 日常启动（以后常用）

**这是以后重启 AutoDL 实验时最常用的一段**，无需重新安装依赖。

```bash
conda activate octo
cd /root/autodl-tmp/AI/octo

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cuda
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
```

可选验证：

```bash
python -c "import jax; print(jax.devices())"
```

## 8. Demo 1：加载 Octo-Small 模型

```python
import jax
from octo.model.octo_model import OctoModel

print('GPU:', jax.devices())
model = OctoModel.load_pretrained('hf://rail-berkeley/octo-small-1.5')
print('Octo-Small loaded successfully!')
print(model.get_pretty_spec())
```

实际输出中的关键信息：

```text
This model is trained with a window size of 2,
predicting 7 dimensional actions 4 steps into the future.
```

即：最多2帧历史图像，单次生成未来4步动作，每步7维。

## 9. Demo 2：图像 + 语言指令 → Action Chunk

### 9.1 输入图像

实验保存的输入图像为：

```text
/root/autodl-tmp/AI/octo/demo_outputs/bridge_input.png
```

它是 256×256 的 RGB 图像。可在 AutoDL JupyterLab 或 VS Code Remote-SSH 中打开。

### 9.2 完整推理代码

运行前先执行第7节的环境变量命令，然后在 Octo 项目根目录执行以下 Python 程序：

```python
import jax
import numpy as np
from PIL import Image
from octo.model.octo_model import OctoModel

model = OctoModel.load_pretrained('hf://rail-berkeley/octo-small-1.5')

# 图像：原始 (256, 256, 3)，扩展 batch 和 history 维度
img = np.array(Image.open('demo_outputs/bridge_input.png').convert('RGB'))
img = img[np.newaxis, np.newaxis, ...]

# 视觉观测以及时间有效性标记
observation = {
    'image_primary': img,
    'timestep_pad_mask': np.array([[True]])
}

# 语言指令
task = model.create_tasks(texts=['pick up the fork'])

# 生成动作，并利用 Bridge 数据集统计量做反归一化
actions = model.sample_actions(
    observation,
    task,
    unnormalization_statistics=model.dataset_statistics['bridge_dataset']['action'],
    rng=jax.random.PRNGKey(0),
)

actions = np.asarray(actions)
print('Image shape:', img.shape)
print('Action shape:', actions.shape)
print('Predicted actions:', actions)
np.save('demo_outputs/octo_actions.npy', actions)
```

本次成功输出：

```text
Image shape: (1, 1, 256, 256, 3)
Action shape: (1, 4, 7)
Saved: demo_outputs/octo_actions.npy
```

动作数组含义：

- `1`：batch size；
- `4`：一次预测未来4步；
- `7`：每一步7个动作分量，主要对应末端位姿增量和夹爪控制（具体坐标方向、单位与夹爪开合约定需要结合数据集定义）。

### 9.3 本次实际输出（节选）

四个时间步的 Z 方向分量约为：

```text
step 1: -0.017946
step 2: -0.010728
step 3: -0.003259
step 4: -0.002246
```

观测到 Z 分量连续为负且幅值逐步减小，**但不能仅凭这条曲线判断抓取任务成功**。

## 10. Demo 3：可视化动作

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

actions = np.load('demo_outputs/octo_actions.npy')[0]
steps = np.arange(1, actions.shape[0] + 1)

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

for i, name in enumerate(['dx', 'dy', 'dz']):
    axes[0].plot(steps, actions[:, i], marker='o', label=name)
axes[0].set_title('End-Effector Translation')
axes[0].legend()
axes[0].grid(True)

for i, name in enumerate(['dRx', 'dRy', 'dRz'], start=3):
    axes[1].plot(steps, actions[:, i], marker='o', label=name)
axes[1].set_title('End-Effector Rotation')
axes[1].legend()
axes[1].grid(True)

for ax in axes:
    ax.set_xticks(steps)
    ax.set_xlabel('Action Step')

plt.tight_layout()
plt.savefig('demo_outputs/octo_actions.png', dpi=150)
print('Saved: demo_outputs/octo_actions.png')
print('Gripper:', actions[:, 6])
```

输出文件：

```text
demo_outputs/bridge_input.png      # 输入图像
demo_outputs/octo_actions.npy      # 预测动作数组
demo_outputs/octo_actions.png      # 平移与旋转曲线
```

## 11. 理论与实验对应

```text
RGB 图像 ──→ 视觉 Tokenizer ───────┐
                                  ├──→ Transformer Backbone
语言指令 ──→ T5 Tokenizer / Encoder ┘             │
                                                 ↓
                                          Readout Embeddings
                                                 │
                                                 ↓
                                        Diffusion Action Head
                                                 │
                                                 ↓
                                     Action Chunk: (1, 4, 7)
```

本次 Demo 使用的是一张静态图片和一句文本指令，**未输入**腕部摄像头图像、第二帧历史观测、显式关节状态或末端位姿；`rng` 为采样随机数种子，`unnormalization_statistics` 用于反归一化，并非额外的视觉观测。

对比 RoboTwin 里的 ACT：两者均可一次生成多个未来动作；Octo 使用条件扩散动作头，而 ACT 使用 CVAE 路线。

## 12. 后续实验建议

1. 保持图像不变，分别输入 `pick up the fork`、`open the drawer`、`move the robot arm to the left`，比较预测动作；
2. 比较不同随机种子下的 Diffusion 动作样本；
3. 结合源码阅读 `OctoModel.sample_actions()` 与 Diffusion Action Head；
4. 后续再接入机器人仿真或真实动作执行，验证任务成功率。

**参考：**

- 官方源码：<https://github.com/octo-models/octo>
- 官方论文：<https://arxiv.org/abs/2405.12213>
- 官方预训练权重：<https://huggingface.co/rail-berkeley/octo-small-1.5>
- 官方推理 Notebook：<https://github.com/octo-models/octo/blob/main/examples/01_inference_pretrained.ipynb>
