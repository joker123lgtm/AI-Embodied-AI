
import math

# 模拟三个视觉 Token 的特征
tokens = {
    "Hammer": [1.0, 0.0, 0.0],
    "Block":  [0.0, 1.0, 0.0],
    "Robot":  [0.0, 0.0, 1.0],
}

# 模拟当前任务查询：重点关注锤子
query = [1.0, 2.0, 0.2]

# 定义点积函数
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

# 计算注意力分数 QK^T / sqrt(d)
scores = {
    name: dot(query, vector) / math.sqrt(3)
    for name, vector in tokens.items()
}

# Softmax：转换成权重
exp_scores = {
    name: math.exp(score)
    for name, score in scores.items()
}
total = sum(exp_scores.values())

weights = {
    name: value / total
    for name, value in exp_scores.items()
}

print("=== Attention Demo ===")
for name, weight in weights.items():
    print(f"{name:8s}: {weight:.1%}  {'#' * round(weight * 40)}")
