# 💣 Grid Duel Arena

![Version](https://img.shields.io/badge/Version-4.1-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-AMP_Enabled-ee4c2c.svg)
![PyGame](https://img.shields.io/badge/PyGame-GUI-yellow.svg)

**Grid Duel Arena** 是一款基于网格对战（类似炸弹人）的强化学习环境与全功能 PvAI 竞技场。项目内置了顶尖的 **Rainbow DQN** 模型，支持玩家与自主训练的 AI 代理进行实时对抗，并提供了直观的实时训练数据可视化仪表盘。

## ✨ 核心特性

### 🧠 深度强化学习 (AI)
* **Rainbow DQN 架构集成**：融合了 C51 分布式强化学习、NoisyNet 探索、Dueling 架构、Double DQN、SumTree-PER（优先经验回放）以及 N-step 收益机制。
* **ResidualAttentionNet**：包含 512 维隐藏层、多头注意力机制（Multi-Head Attention）与残差块，增强了模型对复杂战局的感知能力。
* **BFS 势函数奖励塑形 (Reward Shaping)**：理论安全的寻路与危险规避奖励反馈，加速模型收敛。
* **极致性能优化**：支持 `torch.compile`、CUDA 流处理、混合精度训练 (AMP)，并采用 BatchedEnvWorker 实现多环境并行批量推理。

### 🎮 游戏体验
* **全功能 PvAI 模式**：玩家可直接使用键盘（WASD + J/K）与训练好的模型对战。
* **动态难度切换**：内置 `Easy`、`Normal`、`Hard`、`Expert` 四个难度阶梯，通过调整动作噪声与随机率控制 AI 水平。
* **丰富的道具系统**：威力增强（范围）、速度提升、炸弹容量增加，以及连环爆炸机制。
* **实时监控图表**：对战界面实时渲染 AI 胜率、奖励变化、Epsilon/噪声比例以及 Loss 曲线。

---

## 🛠️ 安装与依赖

推荐使用 Python 3.8+ 版本。请确保系统中已安装以下依赖：

```bash
pip install torch numpy pygame
```

*(注：如果您需要启用 GPU 加速与混合精度训练（AMP），请确保已安装支持 CUDA 版本的 PyTorch。程序启动时会自动检测硬件并生成适配配置。)*

---

## 🚀 快速开始

项目支持三种核心运行模式，可以通过修改启动参数或在 GUI 界面内使用快捷键实时热切换。

### 1. 图形化界面 (GUI) 模式
默认启动 GUI，适用于试玩对战或观察模型表现：
```bash
python main.py
```
* **纯净对战模式** (不进行后台训练)：
  ```bash
  python main.py --play
  ```

### 2. 无头训练 (Headless Training) 模式
适用于在服务器或后台高效挂机训练：
```bash
python main.py --headless --episodes 5000
```

### 3. 高级启动参数
可以通过命令行传递特定参数以覆盖硬件自动分配策略：
```bash
python main.py --device cuda --workers 4 --batch-size 256 --difficulty Hard
```

---

## ⌨️ 操作指南

在 GUI 界面下，您可以使用以下快捷键控制游戏与训练状态：

### 玩家控制
* `W` `A` `S` `D`：移动
* `J`：放置炸弹
* `K`：原地等待 (Stay)

### 模式切换
* `[1]`：PvAI 模式（玩家 vs AI）
* `[2]`：SelfPlay 模式（AI 自我博弈）
* `[3]`：Train 模式（高速并行训练模式）

### 竞技场控制
* `[F1]` - `[F4]`：实时切换 AI 难度（Easy -> Expert）
* `[Space]`：暂停 / 恢复
* `[Up]` / `[Down]`：调整游戏渲染速度
* `[Ctrl + S]`：手动保存模型断点 (Checkpoint)
* `[N]`：回合结束后进入下一局
* `[Esc]`：退出程序

---

## 📂 存档机制
项目会在根目录下自动创建 `grid_duel_ckpt_v4/` 文件夹用于管理模型。
* `model.pth`：当前最新的训练断点。
* `best.pth`：历史上获得最高评估奖励的最佳模型。
* `stats.json`：历史训练数据（胜率、Loss等），用于热重载时恢复图表。
* `strategy_pool.pkl`：策略池，保存多代历史模型，用于提高模型鲁棒性并防范灾难性遗忘。

---
*Developed with Rainbow DQN and PyGame.*