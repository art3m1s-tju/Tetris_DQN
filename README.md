# 基于深度 Q 网络的俄罗斯方块 AI

使用深度强化学习（Deep Q-Network, DQN）训练 AI 自主玩俄罗斯方块。智能体通过与游戏环境的不断交互，自主学习最优落子策略，无需任何人工标注数据。

---

## 效果演示

训练完成后，运行 `test.py` 可生成 AI 对局视频（MP4 格式）。

---

## 项目结构

```
epsilon100percent/
├── train.py                  # 主训练脚本
├── test.py                   # 推理 / 录像脚本
├── src/
│   ├── deep_q_network.py     # 神经网络结构定义
│   └── tetris.py             # 俄罗斯方块游戏环境
├── trained_models/           # 训练过程中保存的模型检查点
│   ├── tetris_10000          # 第 10000 轮检查点
│   ├── tetris_20000
│   ├── tetris_30000
│   ├── tetris_40000
│   ├── tetris_50000
│   └── tetris_60000          # 最终模型
└── tensorboard/              # TensorBoard 训练日志
```

---

## 算法原理

### Deep Q-Network（DQN）

本项目实现了经典 DQN 算法，并加入了以下现代改进：

| 技术 | 说明 |
|------|------|
| **经验回放（Experience Replay）** | 将历史经验存入容量 30,000 的缓冲区，随机采样打破数据相关性 |
| **目标网络（Target Network）** | 独立的目标网络每 1,000 步同步一次，稳定 Q 值估计 |
| **ε-贪心探索（ε-Greedy）** | 探索率从 1.0 线性衰减至 0.01，平衡探索与利用 |
| **梯度裁剪（Gradient Clipping）** | 最大梯度范数限制为 1.0，防止梯度爆炸 |
| **学习率衰减（LR Decay）** | 每 12,000 轮学习率减半，共衰减 5 次 |

### Bellman 方程（训练目标）

$$Q(s, a) \leftarrow r + \gamma \cdot \max_{a'} Q'(s', a')$$

- $Q(s, a)$：主网络预测的当前状态 Q 值
- $Q'(s', a')$：目标网络预测的下一状态 Q 值
- $r$：即时奖励
- $\gamma = 0.99$：折扣因子

---

## 神经网络结构

定义于 [src/deep_q_network.py](src/deep_q_network.py)，为三层全连接网络：

```
输入层：29 维特征向量
    ↓
全连接层 1：Linear(29 → 64) + ReLU
    ↓
全连接层 2：Linear(64 → 64) + ReLU
    ↓
输出层：Linear(64 → 1)  →  Q 值（标量）
```

- **权重初始化**：Xavier Uniform，适合 ReLU 激活函数
- **输出**：单个 Q 值，表示当前动作的预期累计回报

---

## 状态表示（29 维特征）

智能体不直接观察像素，而是接收人工设计的特征向量，大幅提升学习效率：

| 特征组 | 维度 | 说明 |
|--------|------|------|
| 全局统计特征 | 5 | 已消行数、空洞数、凹凸度、总高度、最大高度 |
| 各列高度 | 10 | 棋盘 10 列各自的当前高度 |
| 当前方块 One-Hot | 7 | 当前正在下落的方块类型（7 种） |
| 下一方块 One-Hot | 7 | 即将出现的下一个方块类型（7 种） |

> 特征值直接输入网络，不做归一化处理，依赖 Xavier 初始化和网络自适应学习。

---

## 奖励函数

定义于 [src/tetris.py](src/tetris.py)：

```python
reward = 1 + (lines_cleared ** 2) * board_width   # 基础奖励 + 消行奖励（二次方缩放）
```

| 事件 | 奖励值 |
|------|--------|
| 每步存活 | +1 |
| 消 1 行 | +10 |
| 消 2 行 | +40 |
| 消 3 行 | +90 |
| 消 4 行（Tetris）| +160 |
| 产生新空洞 | -0.1 / 个 |
| 游戏结束 | -2 |

二次方奖励设计鼓励 AI 优先追求多行同消（Tetris 策略），而非单行消除。

---

## 训练超参数

| 参数 | 值 |
|------|----|
| 总训练轮数 | 60,000 |
| Batch Size | 512 |
| 初始学习率 | 0.001 |
| 优化器 | Adam |
| 学习率衰减步长 | 每 12,000 轮 × 0.5 |
| 折扣因子 γ | 0.99 |
| 初始探索率 ε | 1.0 |
| 最终探索率 ε | 0.01 |
| ε 衰减周期 | 60,000 轮 |
| 经验回放容量 | 30,000 |
| 目标网络更新间隔 | 每 1,000 轮 |
| 预热步数（Warmup）| 3,000 步 |
| 损失函数 | SmoothL1Loss（Huber Loss）|
| 梯度裁剪范数 | 1.0 |
| 模型保存间隔 | 每 10,000 轮 |

---

## 游戏环境

定义于 [src/tetris.py](src/tetris.py)，实现完整的俄罗斯方块逻辑：

- **棋盘尺寸**：10 列 × 20 行
- **方块种类**：7 种标准方块（O、T、S、Z、I、L、J）
- **动作空间**：`(x 位置, 旋转次数)` 元组，智能体选择落点列和旋转角度，方块自动下落
- **出块机制**：随机袋（Random Bag），保证每 7 个方块中每种各出现一次
- **渲染**：使用 OpenCV 生成可视化画面，支持录制 MP4 视频

---

## 快速开始

### 环境依赖

```bash
pip install torch torchvision opencv-python tensorboard numpy
```

### 训练模型

```bash
python train.py
```

训练过程中模型每 10,000 轮自动保存至 `trained_models/`，日志写入 `tensorboard/`。

### 查看训练曲线

```bash
tensorboard --logdir tensorboard/
```

### 测试 / 生成对局视频

```bash
python test.py
```

视频将以时间戳命名保存至项目根目录（如 `output_20260226_192106.mp4`）。

---

## 训练监控指标

TensorBoard 记录以下指标：

| 指标 | 说明 |
|------|------|
| Score | 每局得分 |
| Pieces | 每局放置方块数 |
| Lines Cleared | 每局消行数 |
| Loss | 训练损失 |
| Epsilon | 当前探索率 |
| Replay Memory Size | 经验回放缓冲区大小 |
| Learning Rate | 当前学习率 |

---

## 参考

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — DQN 原始论文（Mnih et al., 2013）
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — DQN Nature 版本（Mnih et al., 2015）
- 原始 Tetris 环境实现参考：[Viet Nguyen](https://github.com/uvipen/Tetris-deep-Q-learning-pytorch)
