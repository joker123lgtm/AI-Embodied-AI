import jax
import numpy as np
from PIL import Image
from octo.model.octo_model import OctoModel

# 1. 加载 Octo-Small
model = OctoModel.load_pretrained(
    "hf://rail-berkeley/octo-small-1.5"
)

# 2. 加载同一张输入图像
img = np.array(
    Image.open("../demo_outputs/bridge_input.png").convert("RGB")
)

img = img[np.newaxis, np.newaxis, ...]

observation = {
    "image_primary": img,
    "timestep_pad_mask": np.array([[True]])
}

# 3. 定义三条不同的语言指令
instructions = [
    "pick up the fork",
    "open the drawer",
    "move the robot arm to the left"
]

results = []

# 4. 分别执行推理
for i, instruction in enumerate(instructions):

    task = model.create_tasks(
        texts=[instruction]
    )

    actions = model.sample_actions(
        observation,
        task,
        unnormalization_statistics=model.dataset_statistics[
            "bridge_dataset"
        ]["action"],
        rng=jax.random.PRNGKey(42)
    )

    actions = np.asarray(actions)[0]
    results.append(actions)

    print(f"\n===== Task {i+1}: {instruction} =====")
    print("Shape:", actions.shape)
    print(np.round(actions, 4))

# 5. 保存三组动作
results = np.stack(results)

np.save(
    "../demo_outputs/language_comparison.npy",
    results
)

print("\nSaved: demo_outputs/language_comparison.npy")