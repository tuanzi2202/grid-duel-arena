# 💣 Grid Duel Arena v1.0
**Biomimetic Lightweight AI Combat Trainer** *(炸弹对战竞技场：玩家 vs 自我进化 AI)*

一款基于 Pygame 与 PyTorch 构建的纯逻辑极简强化学习环境。本项目旨在证明：**即便是在纯 CPU、2GB 内存的“垃圾配置”古董电脑上，依然能够流畅跑通并训练一套高效的深度强化学习大逃杀 AI。**

## ✨ 核心亮点 / Features

本项目的 AI 抛弃了重型网络（无世界模型、无 VAE 重建开销），转而采用**仿生学**启发的高效架构：

- **🪳 蟑螂反射弧 (Roach Reflex Arc Dual Channel)**
  模拟昆虫神经递质的双通道决策网络：
  - `Fast Channel (快速反射通道)`：处理近场 18 维危险感知（躲避炸弹、遇障寻路）。
  - `Slow Channel (慢速策略通道)`：处理全局特征（血量优势、进攻路线规划、吃道具）。
  - 最终通过注意力门控 (Gate) 动态融合两种决策权重。
- **🧬 免疫克隆选择 (Immune Clone Selection)**
  受生物免疫系统启发的策略种群管理：
  - 定期克隆当前最优网络作为“记忆细胞”（历史策略池）。
  - 在 Self-Play 或 Train 模式下，随机抽取历史策略作为对手，有效防止策略坍缩 (Mode Collapse)。
- **⚡ 极轻量级优先经验回放 (Light PER)**
  专门针对纯 CPU 环境优化的 `O(log N)` 轻量级优先经验回放机制，结合 n-step Double DQN 极速收敛。

## 🛠️ 安装指南 / Installation

只需标准的 Python 3 环境即可运行。

```bash
# 1. 克隆代码仓库
git clone [https://github.com/yourusername/grid-duel-arena.git](https://github.com/yourusername/grid-duel-arena.git)
cd grid-duel-arena

# 2. 安装必要依赖包 (极少依赖)
pip install torch pygame numpy
```

## 🚀 快速开始 / Quick Start

直接运行主入口文件：

```bash
python grid_duel_arena.py
```

### 🎮 模式与快捷键 (Hotkeys)

游戏启动后，你可以通过以下快捷键随时无缝切换运行状态：

**全局控制：**
- `[1]` : **PvAI 模式** - 玩家亲自下场与当前 AI 决斗。
- `[2]` : **Self-Play 模式** - AI 与历史策略池中的自己对战（中等倍速）。
- `[3]` : **Fast Train 模式** - 后台极速演算训练（最高倍速，无视帧率限制）。
- `[Space]` : 暂停/继续
- `[↑] / [↓]` : 加速 / 减速全局运算
- `[S]` : 手动存档权重到 `grid_duel_ckpt` 目录

**PvAI 模式玩家操作：**
- `W / A / S / D` : 上下左右移动
- `J` : 放置炸弹 💣
- `K` : 原地停留等待

## 📊 监控面板说明

右侧提供了实时的网络状态监控，包含三大迷你动态图表：
- **AI WinRate%**：近期胜率滑动平均。
- **AI Reward**：环境奖励塑形得分（不仅包含胜负，还包含如“靠近玩家”、“远离火海”等连续型生存奖励）。
- **Epsilon**：当前探索率（$\epsilon$-greedy 的衰减状态）。

## 📁 存档结构

训练会自动生成 `grid_duel_ckpt/` 文件夹，包含：
- `model.pth`: 最新检查点。
- `best.pth`: 历史最高分的模型。
- `strategy_pool.pkl`: 免疫克隆选择的策略池。
- `stats.json`: 胜率与 Loss 等统计图表数据。

## 📜 License
MIT License. 自由把玩，欢迎提 PR！