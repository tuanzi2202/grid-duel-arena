#!/usr/bin/env python3
"""
Grid Duel Arena v1.0 — Biomimetic Lightweight AI Combat Trainer
===============================================================
炸弹对战竞技场：玩家 vs 自我进化AI
设计目标：垃圾配置电脑（纯CPU、2GB内存）也能流畅训练

核心仿生架构：
  🪳蟑螂反射弧：双通道决策（快速反射+ 慢速策略）
  🧬 免疫克隆选择：策略种群淘汰 + 精英保留
  ⚡ 极轻量DQN：无世界模型、无VAE重建开销

pip install torch pygame numpy
"""

import pygame
import random
import math
import numpy as np
import os
import sys
import time
import json
import copy
import pickle
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#╔═══════════════════════════════════════╗
# ║        全局配置 (低配友好)           ║
# ╚═══════════════════════════════════════╝

VERSION = "1.0"
DEVICE = torch.device("cpu")  # 强制CPU，兼容所有垃圾配置

# 竞技场参数
ARENA_COLS = 13  # 奇数保证对称
ARENA_ROWS = 11
CELL = 40  # 像素/格
ARENA_W = ARENA_COLS * CELL
ARENA_H = ARENA_ROWS * CELL

# 窗口布局
PANEL_W = 260
BOTTOM_H = 80
WIN_W = ARENA_W + PANEL_W
WIN_H = ARENA_H + BOTTOM_H

# AI 参数 (极轻量)
STATE_DIM = 42  # 特征向量维度
ACTION_DIM = 6  # 上下左右 + 放炸弹 + 静止
HIDDEN_DIM = 128  # 隐藏层（比Dreamer小10倍）
BATCH_SIZE = 64  # 小batch，CPU友好
MEMORY_SIZE = 15000
GAMMA = 0.97
LR = 5e-4
TAU = 0.01
EPS_START = 1.0
EPS_END = 0.03
EPS_DECAY = 5000
N_STEP = 3

# 游戏规则
MAX_HP = 3
BOMB_TIMER = 8  # 炸弹倒计时（帧数）
BOMB_RANGE = 2  # 爆炸范围
BOMB_COOLDOWN = 12  # 放炸弹冷却
POWERUP_INTERVAL = 60  # 道具刷新间隔
MAX_ROUND_STEPS = 500
BASE_FPS = 8

# 存档
CKPT_DIR = "grid_duel_ckpt"
CKPT_MODEL = os.path.join(CKPT_DIR, "model.pth")
CKPT_BEST = os.path.join(CKPT_DIR, "best.pth")
CKPT_POOL = os.path.join(CKPT_DIR, "strategy_pool.pkl")
CKPT_STATS = os.path.join(CKPT_DIR, "stats.json")

# 方向映射
DIR_MAP = {
    0: (0, -1),   # Up
    1: (0, 1),    # Down
    2: (-1, 0),   # Left
    3: (1, 0),    # Right
    4: (0, 0),    # Bomb (原地放)
    5: (0, 0),    # Stay
}
DIR_NAMES = ["↑", "↓", "←", "→", "💣", "·"]

# ╔═══════════════════════════════════════╗
# ║             颜色主题                  ║
# ╚═══════════════════════════════════════╝

C_BG        = (12, 12, 22)
C_GRID      = (25, 25, 40)
C_WALL      = (60, 65, 80)
C_BRICK     = (140, 100, 60)
C_BRICK_D   = (110, 75, 40)
C_FLOOR     = (18, 18, 30)
C_PANEL     = (16, 16, 28)
C_BOTTOM    = (22, 22, 38)
C_TEXT      = (180, 180, 210)
C_DIM       = (100, 100, 130)
C_GOOD      = (60, 255, 140)
C_BAD       = (255, 70, 70)
C_WARN      = (255, 180, 40)
C_HIGHLIGHT = (255, 210, 50)

C_PLAYER    = (0, 200, 255)
C_PLAYER_D  = (0, 140, 200)
C_AI        = (255, 80, 80)
C_AI_D      = (200, 50, 50)

C_BOMB      = (255, 220, 50)
C_BOMB_FUSE = (255, 100, 30)
C_EXPLODE   = [(255, 255, 200), (255, 200, 50), (255, 120, 20), (200, 50, 0)]
C_POWERUP   = (100, 255, 200)
C_SHIELD    = (80, 160, 255)

C_CHART_WIN = (80, 200, 255)
C_CHART_REW = (255, 175, 55)
C_CHART_EPS = (200, 100, 255)


# ╔═══════════════════════════════════════╗
# ║           地图生成器                  ║
# ╚═══════════════════════════════════════╝

# 格子类型
EMPTY = 0
WALL = 1       # 不可摧毁
BRICK = 2      # 可摧毁
POWERUP_CELL = 3

def generate_arena(seed=None):
    """生成对称竞技场地图"""
    rng = random.Random(seed)
    grid = [[EMPTY] * ARENA_COLS for _ in range(ARENA_ROWS)]

    # 边界墙
    for x in range(ARENA_COLS):
        grid[0][x] = WALL
        grid[ARENA_ROWS - 1][x] = WALL
    for y in range(ARENA_ROWS):
        grid[y][0] = WALL
        grid[y][ARENA_COLS - 1] = WALL

    # 内部柱子 (经典炸弹人布局：偶数行偶数列放柱子)
    for y in range(2, ARENA_ROWS - 2, 2):
        for x in range(2, ARENA_COLS - 2, 2):
            grid[y][x] = WALL

    # 随机砖块 (对称放置)
    half_x = ARENA_COLS // 2
    for y in range(1, ARENA_ROWS - 1):
        for x in range(1, half_x + 1):
            if grid[y][x] != EMPTY:
                continue
            mx = ARENA_COLS - 1 - x  # 镜像x
            if grid[y][mx] != EMPTY:
                continue
            # 出生点保护区
            if (y <= 2 and x <= 2) or (y <= 2 and mx >= ARENA_COLS - 3):
                continue
            if (y >= ARENA_ROWS - 3 and x <= 2) or (y >= ARENA_ROWS - 3 and mx >= ARENA_COLS - 3):
                continue
            if rng.random() < 0.35:
                grid[y][x] = BRICK
                grid[y][mx] = BRICK

    return grid


# ╔═══════════════════════════════════════╗
# ║         游戏核心逻辑                  ║
# ╚═══════════════════════════════════════╝

class Bomb:
    __slots__ = ['x', 'y', 'timer', 'owner', 'power']
    def __init__(self, x, y, owner, power=BOMB_RANGE):
        self.x = x
        self.y = y
        self.timer = BOMB_TIMER
        self.owner = owner
        self.power = power


class Explosion:
    __slots__ = ['x', 'y', 'timer']
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.timer = 4


class PowerUp:
    __slots__ = ['x', 'y', 'kind']
    # kind: 0=range+1, 1=speed+1, 2=extra_bomb
    def __init__(self, x, y, kind):
        self.x = x
        self.y = y
        self.kind = kind


class Fighter:
    def __init__(self, x, y, is_ai=False):
        self.x = x
        self.y = y
        self.hp = MAX_HP
        self.max_hp = MAX_HP
        self.bomb_cooldown = 0
        self.bomb_power = BOMB_RANGE
        self.max_bombs = 1
        self.active_bombs = 0
        self.speed = 1           # 未使用，预留
        self.invincible = 0      # 无敌帧数（被炸后短暂无敌）
        self.is_ai = is_ai
        self.score = 0
        self.kills = 0
        self.deaths = 0
        self.last_action = 5

    def reset(self, x, y):
        self.x = x
        self.y = y
        self.hp = self.max_hp
        self.bomb_cooldown = 0
        self.active_bombs = 0
        self.invincible = 0
        self.last_action = 5


class DuelArena:
    """纯逻辑游戏世界，无渲染依赖"""

    def __init__(self, seed=None):
        self.seed = seed
        self.grid = []
        self.player = Fighter(1, 1, is_ai=False)
        self.ai = Fighter(ARENA_COLS - 2, ARENA_ROWS - 2, is_ai=True)
        self.bombs = []
        self.explosions = []
        self.powerups = []
        self.step_count = 0
        self.round_over = False
        self.winner = None       # 'player', 'ai', 'draw', None
        self.powerup_timer = POWERUP_INTERVAL
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.grid = generate_arena(self.seed)
        self.player.reset(1, ARENA_ROWS - 2)
        self.ai.reset(ARENA_COLS - 2, 1)
        self.bombs.clear()
        self.explosions.clear()
        self.powerups.clear()
        self.step_count = 0
        self.round_over = False
        self.winner = None
        self.powerup_timer = POWERUP_INTERVAL
        return self.get_state(for_ai=True)

    def _can_move(self, x, y):
        if x < 0 or x >= ARENA_COLS or y < 0 or y >= ARENA_ROWS:
            return False
        if self.grid[y][x] in (WALL, BRICK):
            return False
        #炸弹阻挡
        for b in self.bombs:
            if b.x == x and b.y == y:
                return False
        return True

    def _do_move(self, fighter, action):
        """执行一个角色的动作"""
        fighter.last_action = action
        if fighter.bomb_cooldown > 0:
            fighter.bomb_cooldown -= 1
        if fighter.invincible > 0:
            fighter.invincible -= 1

        if action == 4:  # 放炸弹
            if fighter.bomb_cooldown <= 0 and fighter.active_bombs < fighter.max_bombs:
                # 检查当前位置没有炸弹
                has_bomb = any(b.x == fighter.x and b.y == fighter.y for b in self.bombs)
                if not has_bomb:
                    self.bombs.append(
                        Bomb(fighter.x, fighter.y, fighter, fighter.bomb_power))
                    fighter.active_bombs += 1
                    fighter.bomb_cooldown = BOMB_COOLDOWN
        elif action < 4:  # 移动
            dx, dy = DIR_MAP[action]
            nx, ny = fighter.x + dx, fighter.y + dy
            if self._can_move(nx, ny):
                # 不能走到对方位置
                other = self.ai if not fighter.is_ai else self.player
                if not (nx == other.x and ny == other.y):
                    fighter.x = nx
                    fighter.y = ny

    def _explode_bomb(self, bomb):
        """炸弹爆炸，返回爆炸格列表"""
        cells = [(bomb.x, bomb.y)]
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in dirs:
            for r in range(1, bomb.power + 1):
                ex, ey = bomb.x + dx * r, bomb.y + dy * r
                if ex < 0 or ex >= ARENA_COLS or ey < 0 or ey >= ARENA_ROWS:
                    break
                if self.grid[ey][ex] == WALL:
                    break
                cells.append((ex, ey))
                if self.grid[ey][ex] == BRICK:
                    #砖块被摧毁
                    self.grid[ey][ex] = EMPTY
                    # 概率掉落道具
                    if self.rng.random() < 0.25:
                        kind = self.rng.randint(0, 2)
                        self.powerups.append(PowerUp(ex, ey, kind))
                    break
        return cells

    def _check_explosion_hit(self, cells):
        """检查爆炸是否击中角色"""
        for cx, cy in cells:
            for fighter in [self.player, self.ai]:
                if fighter.x == cx and fighter.y == cy and fighter.invincible <= 0:
                    fighter.hp -= 1
                    fighter.invincible = 6  # 短暂无敌
                    if fighter.hp <= 0:
                        other = self.ai if fighter is self.player else self.player
                        other.kills += 1
                        fighter.deaths += 1

    def _check_powerups(self):
        """检查道具拾取"""
        for fighter in [self.player, self.ai]:
            i = 0
            while i < len(self.powerups):
                pu = self.powerups[i]
                if pu.x == fighter.x and pu.y == fighter.y:
                    if pu.kind == 0:  # 爆炸范围+1
                        fighter.bomb_power = min(fighter.bomb_power + 1, 5)
                    elif pu.kind == 1:  # 速度（预留）
                        pass
                    elif pu.kind == 2:  # 额外炸弹
                        fighter.max_bombs = min(fighter.max_bombs + 1, 3)
                    self.powerups.pop(i)
                else:
                    i += 1

    def step(self, player_action, ai_action):
        """
        同时执行双方动作
        返回: (ai_state, ai_reward, done)
        """
        if self.round_over:
            return self.get_state(for_ai=True), 0.0, True

        self.step_count += 1

        # 1. 执行动作
        self._do_move(self.player, player_action)
        self._do_move(self.ai, ai_action)

        # 2. 炸弹计时器
        exploded_cells = []
        i = 0
        while i < len(self.bombs):
            self.bombs[i].timer -= 1
            if self.bombs[i].timer <= 0:
                cells = self._explode_bomb(self.bombs[i])
                exploded_cells.extend(cells)
                for cx, cy in cells:
                    self.explosions.append(Explosion(cx, cy))
                # 链爆：检查爆炸范围内的其他炸弹
                j = 0
                while j < len(self.bombs):
                    if j != i:
                        for cx, cy in cells:
                            if self.bombs[j].x == cx and self.bombs[j].y == cy:
                                self.bombs[j].timer = 0
                    j += 1
                self.bombs[i].owner.active_bombs -= 1
                self.bombs.pop(i)
            else:
                i += 1

        # 3. 爆炸伤害
        self._check_explosion_hit(exploded_cells)

        # 4. 更新爆炸动画
        i = 0
        while i < len(self.explosions):
            self.explosions[i].timer -= 1
            if self.explosions[i].timer <= 0:
                self.explosions.pop(i)
            else:
                i += 1

        # 5. 道具拾取
        self._check_powerups()

        # 6. 定期刷新道具
        self.powerup_timer -= 1
        if self.powerup_timer <= 0:
            self.powerup_timer = POWERUP_INTERVAL
            self._spawn_random_powerup()

        # 7. 胜负判定
        ai_reward = 0.0
        done = False

        if self.player.hp <= 0 and self.ai.hp <= 0:
            self.winner = 'draw'
            self.round_over = True
            done = True
            ai_reward = 0.0
        elif self.player.hp <= 0:
            self.winner = 'ai'
            self.round_over = True
            done = True
            ai_reward = 20.0
        elif self.ai.hp <= 0:
            self.winner = 'player'
            self.round_over = True
            done = True
            ai_reward = -20.0
        elif self.step_count >= MAX_ROUND_STEPS:
            # 超时：HP多的赢
            if self.ai.hp > self.player.hp:
                self.winner = 'ai'
                ai_reward = 10.0
            elif self.player.hp > self.ai.hp:
                self.winner = 'player'
                ai_reward = -10.0
            else:
                self.winner = 'draw'
                ai_reward = -2.0
            self.round_over = True
            done = True

        if not done:
            # 生存奖励塑形
            ai_reward = self._compute_shaping_reward()

        return self.get_state(for_ai=True), ai_reward, done

    def _compute_shaping_reward(self):
        """奖励塑形（蟑螂反射弧启发：靠近食物=正，靠近危险=负）"""
        reward = 0.0
        # 微小的存活奖励
        reward += 0.01

        # 靠近玩家（鼓励进攻）
        dist = abs(self.ai.x - self.player.x) + abs(self.ai.y - self.player.y)
        if dist <= 3:
            reward += 0.05
        elif dist >= 8:
            reward -= 0.02

        # 远离爆炸（鼓励生存）
        for exp in self.explosions:
            if abs(self.ai.x - exp.x) + abs(self.ai.y - exp.y) <= 1:
                reward -= 0.3

        # 远离自己炸弹（鼓励不自杀）
        for b in self.bombs:
            if abs(self.ai.x - b.x) + abs(self.ai.y - b.y) <= 1 and b.timer <= 3:
                reward -= 0.15

        # HP差距奖励
        hp_diff = self.ai.hp - self.player.hp
        reward += hp_diff * 0.05

        return reward

    def _spawn_random_powerup(self):
        """在随机空地生成道具"""
        empties = []
        occupied = {(self.player.x, self.player.y), (self.ai.x, self.ai.y)}
        for b in self.bombs:
            occupied.add((b.x, b.y))
        for pu in self.powerups:
            occupied.add((pu.x, pu.y))
        for y in range(1, ARENA_ROWS - 1):
            for x in range(1, ARENA_COLS - 1):
                if self.grid[y][x] == EMPTY and (x, y) not in occupied:
                    empties.append((x, y))
        if empties and len(self.powerups) < 4:
            x, y = self.rng.choice(empties)
            kind = self.rng.randint(0, 2)
            self.powerups.append(PowerUp(x, y, kind))

    def get_state(self, for_ai=True):
        """
        构建状态向量（蟑螂反射弧启发：分为"近场感知"和"全局态势"）
        """
        me = self.ai if for_ai else self.player
        enemy = self.player if for_ai else self.ai

        features = []

        # — 自身状态 (6维) —
        features.append(me.x / ARENA_COLS)
        features.append(me.y / ARENA_ROWS)
        features.append(me.hp / me.max_hp)
        features.append(me.bomb_cooldown / BOMB_COOLDOWN)
        features.append(me.active_bombs / max(me.max_bombs, 1))
        features.append(me.invincible / 6.0)

        # — 敌人状态 (5维) —
        features.append((enemy.x - me.x) / ARENA_COLS)
        features.append((enemy.y - me.y) / ARENA_ROWS)
        features.append(enemy.hp / enemy.max_hp)
        dist = abs(enemy.x - me.x) + abs(enemy.y - me.y)
        features.append(dist / (ARENA_COLS + ARENA_ROWS))
        angle = math.atan2(enemy.y - me.y, enemy.x - me.x)
        features.append(angle / math.pi)

        # — 四方向近场感知（蟑螂触角）(12维) —
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in dirs:
            # 到墙/砖距离
            wall_dist = 0
            for r in range(1, max(ARENA_COLS, ARENA_ROWS)):
                cx, cy = me.x + dx * r, me.y + dy * r
                if cx < 0 or cx >= ARENA_COLS or cy < 0 or cy >= ARENA_ROWS:
                    break
                if self.grid[cy][cx] in (WALL, BRICK):
                    break
                wall_dist = r
            features.append(wall_dist / max(ARENA_COLS, ARENA_ROWS))

            # 该方向有无炸弹危险
            danger = 0.0
            for b in self.bombs:
                if dx != 0 and b.y == me.y and 0 < (b.x - me.x) * dx <= b.power + 1:
                    danger = max(danger, 1.0 - b.timer / BOMB_TIMER)
                if dy != 0 and b.x == me.x and 0 < (b.y - me.y) * dy <= b.power + 1:
                    danger = max(danger, 1.0 - b.timer / BOMB_TIMER)
            features.append(danger)

            # 该方向有无爆炸
            has_exp = 0.0
            for exp in self.explosions:
                if dx != 0 and exp.y == me.y and 0 < (exp.x - me.x) * dx <= 2:
                    has_exp = 1.0
                if dy != 0 and exp.x == me.x and 0 < (exp.y - me.y) * dy <= 2:
                    has_exp = 1.0
            features.append(has_exp)

        # — 炸弹全局信息 (6维) —
        my_bomb_count = sum(1 for b in self.bombs if b.owner is me)
        enemy_bomb_count = sum(1 for b in self.bombs if b.owner is enemy)
        features.append(my_bomb_count / 3.0)
        features.append(enemy_bomb_count / 3.0)

        # 最近炸弹
        min_bomb_dist = 1.0
        min_bomb_timer = 1.0
        for b in self.bombs:
            bd = (abs(b.x - me.x) + abs(b.y - me.y)) / (ARENA_COLS + ARENA_ROWS)
            if bd < min_bomb_dist:
                min_bomb_dist = bd
                min_bomb_timer = b.timer / BOMB_TIMER
        features.append(min_bomb_dist)
        features.append(min_bomb_timer)

        # 当前位置是否在爆炸范围内
        in_danger = 0.0
        for b in self.bombs:
            if (b.x == me.x and abs(b.y - me.y) <= b.power) or \
               (b.y == me.y and abs(b.x - me.x) <= b.power):
                in_danger = max(in_danger, 1.0 - b.timer / BOMB_TIMER)
        features.append(in_danger)

        # 是否被围困（四方向都不能动）
        stuck = 1.0
        for dx, dy in dirs:
            if self._can_move(me.x + dx, me.y + dy):
                stuck = 0.0
                break
        features.append(stuck)

        # — 道具信息 (3维) —
        min_pu_dist = 1.0
        min_pu_dx = 0.0
        min_pu_dy = 0.0
        for pu in self.powerups:
            pd = (abs(pu.x - me.x) + abs(pu.y - me.y)) / (ARENA_COLS + ARENA_ROWS)
            if pd < min_pu_dist:
                min_pu_dist = pd
                min_pu_dx = (pu.x - me.x) / ARENA_COLS
                min_pu_dy = (pu.y - me.y) / ARENA_ROWS
        features.append(min_pu_dist)
        features.append(min_pu_dx)
        features.append(min_pu_dy)

        # — 局势(4维) —
        features.append(self.step_count / MAX_ROUND_STEPS)
        features.append(me.bomb_power / 5.0)
        features.append(me.max_bombs / 3.0)
        features.append(len(self.powerups) / 4.0)

        #补齐到 STATE_DIM
        while len(features) < STATE_DIM:
            features.append(0.0)

        return np.array(features[:STATE_DIM], dtype=np.float32)

    def get_danger_map(self):
        """返回危险热力图 (用于可视化)"""
        dmap = np.zeros((ARENA_ROWS, ARENA_COLS), dtype=np.float32)
        for b in self.bombs:
            urgency = 1.0 - b.timer / BOMB_TIMER
            dmap[b.y][b.x] = max(dmap[b.y][b.x], urgency)
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                for r in range(1, b.power + 1):
                    ex, ey = b.x + dx * r, b.y + dy * r
                    if 0 <= ex < ARENA_COLS and 0 <= ey < ARENA_ROWS:
                        if self.grid[ey][ex] == WALL:
                            break
                        dmap[ey][ex] = max(dmap[ey][ex], urgency * 0.7)
                    else:
                        break
        for exp in self.explosions:
            dmap[exp.y][exp.x] = 1.0
        return dmap


#╔═══════════════════════════════════════╗
# ║      神经网络（蟑螂反射弧架构）        ║
# ╚═══════════════════════════════════════╝

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ReflexNet(nn.Module):
    """
    蟑螂反射弧启发的双通道网络：
    - 快速通道(Fast)：3层小网络，处理近场危险感知（逃跑）
    - 慢速通道(Slow)：5层网络，处理全局策略（进攻/道具）
    最终融合输出
    """
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        # 快速反射通道（仅用近场特征: 12维方向感知 + 6维炸弹信息 = 18维）
        self.fast_net = nn.Sequential(
            nn.Linear(18, 48),
            Swish(),
            nn.Linear(48, 24),
            Swish(),
            nn.Linear(24, action_dim)
        )

        # 慢速策略通道（全部特征）
        self.slow_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            Swish(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            Swish(),
            nn.Linear(hidden, hidden // 2),
            Swish(),
            nn.Linear(hidden // 2, action_dim),
        )

        # Dueling: Value + Advantage 融合
        self.value_head = nn.Sequential(
            nn.Linear(action_dim * 2, 32),
            Swish(),
            nn.Linear(32, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(action_dim * 2, 32),
            Swish(),
            nn.Linear(32, action_dim),
        )

        # 门控权重（学习快慢通道的融合比例）
        self.gate = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 拆分近场特征（index 11~28是四方向感知 + 炸弹信息）
        fast_feat = x[:, 11:29] if x.shape[-1] >= 29 else x[:, :18]
        fast_q = self.fast_net(fast_feat)
        slow_q = self.slow_net(x)

        # 门控融合
        gate = self.gate(x)  # [batch, 1]
        combined = torch.cat([fast_q * gate, slow_q * (1 - gate)], dim=-1)

        v = self.value_head(combined)
        a = self.advantage_head(combined)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q

    def get_gate_value(self, x):
        """返回门控值，用于可视化"""
        with torch.no_grad():
            return self.gate(x).item()


# ╔═══════════════════════════════════════╗
# ║      经验回放 (轻量PER)               ║
# ╚═══════════════════════════════════════╝

class LightPER:
    """轻量级优先经验回放，CPU友好"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, transition):
        mx = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(transition)
        self.priorities.append(mx)

    def sample(self, batch_size, beta=0.4):
        n = len(self.buffer)
        p = np.array(self.priorities, dtype=np.float64)
        p = p ** self.alpha
        p /= p.sum()
        size = min(batch_size, n)
        idx = np.random.choice(n, size, p=p, replace=False)
        samples = [self.buffer[i] for i in idx]
        w = (n * p[idx]) ** (-beta)
        w /= w.max()
        return samples, idx, torch.tensor(w, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            if 0 <= i < len(self.priorities):
                self.priorities[i] = abs(td) + 1e-6

    def __len__(self):
        return len(self.buffer)


class NStepBuffer:
    def __init__(self, n=N_STEP, gamma=GAMMA):
        self.n = n
        self.gamma = gamma
        self.buffer = deque(maxlen=n)

    def push(self, transition):
        self.buffer.append(transition)

    def get(self):
        if len(self.buffer) < self.n:
            return None
        s0, a0 = self.buffer[0][0], self.buffer[0][1]
        r = sum(self.gamma ** i * self.buffer[i][2] for i in range(self.n))
        return (s0, a0, r, self.buffer[-1][3], self.buffer[-1][4])

    def flush(self):
        results = []
        while self.buffer:
            s0, a0 = self.buffer[0][0], self.buffer[0][1]
            r = sum(self.gamma ** i * self.buffer[i][2] for i in range(len(self.buffer)))
            results.append((s0, a0, r, self.buffer[-1][3], self.buffer[-1][4]))
            self.buffer.popleft()
        return results

    def reset(self):
        self.buffer.clear()


# ╔═══════════════════════════════════════╗
# ║策略池（免疫克隆选择机制）             ║
# ╚═══════════════════════════════════════╝

class StrategyPool:
    """
    免疫系统启发的策略种群管理：
    - 定期将当前网络克隆为"记忆细胞"
    - 训练时随机抽取历史策略作为对手（防止策略坍缩）
    - 淘汰弱策略，保留精英
    """
    def __init__(self, max_size=8):
        self.max_size = max_size
        self.pool = []     # [(name, state_dict, fitness)]
        self.generation = 0

    def add(self, name, state_dict, fitness):
        self.pool.append((name, copy.deepcopy(state_dict), fitness))
        self.generation += 1
        #淘汰：保留前max_size个
        if len(self.pool) > self.max_size:
            self.pool.sort(key=lambda x: x[2], reverse=True)
            self.pool = self.pool[:self.max_size]

    def sample_opponent(self):
        """随机抽取一个历史策略"""
        if not self.pool:
            return None
        return random.choice(self.pool)

    def best(self):
        if not self.pool:
            return None
        return max(self.pool, key=lambda x: x[2])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"pool": self.pool, "gen": self.generation}, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.pool = data.get("pool", [])
            self.generation = data.get("gen", 0)
            print(f"  🧬 Loaded strategy pool: {len(self.pool)} strategies, gen {self.generation}")


# ╔═══════════════════════════════════════╗
# ║             迷你图表                  ║
# ╚═══════════════════════════════════════╝

class MiniChart:
    def __init__(self, x, y, w, h, title, color, max_pts=150):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.title = title
        self.color = color
        self.data = deque(maxlen=max_pts)

    def add(self, v):
        self.data.append(v)

    def draw(self, surf, font):
        pygame.draw.rect(surf, (10, 10, 22), (self.x, self.y, self.w, self.h))
        pygame.draw.rect(surf, (40, 40, 60), (self.x, self.y, self.w, self.h), 1)
        surf.blit(font.render(self.title, True, C_DIM), (self.x + 4, self.y + 2))
        if len(self.data) < 2:
            return
        dl = list(self.data)
        mn, mx = min(dl), max(dl)
        rng = mx - mn if mx != mn else 1.0
        cy, ch = self.y + 15, self.h - 18
        # 当前值
        surf.blit(font.render(f"{dl[-1]:.1f}", True, self.color),
                  (self.x + self.w - 48, self.y + 2))
        # 移动平均线
        win = min(30, len(dl))
        avg = []
        for i in range(len(dl)):
            s = max(0, i - win + 1)
            avg.append(sum(dl[s:i + 1]) / (i - s + 1))
        pts = []
        apts = []
        n = len(dl)
        for i in range(n):
            px = self.x + 2 + (self.w - 4) * i / max(n - 1, 1)
            py1 = cy + ch - ch * (dl[i] - mn) / rng
            py1 = max(cy, min(cy + ch, py1))
            pts.append((px, py1))
            py2 = cy + ch - ch * (avg[i] - mn) / rng
            py2 = max(cy, min(cy + ch, py2))
            apts.append((px, py2))
        if len(pts) >= 2:
            pygame.draw.lines(surf, tuple(v // 3 for v in self.color), False, pts, 1)
        if len(apts) >= 2:
            pygame.draw.lines(surf, self.color, False, apts, 2)


# ╔═══════════════════════════════════════╗
# ║             粒子系统                  ║
# ╚═══════════════════════════════════════╝

class Particle:
    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'max_life', 'color', 'size']
    def __init__(self, x, y, color, speed_range=(1, 4)):
        self.x, self.y = x, y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(*speed_range)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = random.uniform(0.3, 0.7)
        self.max_life = self.life
        self.color = color
        self.size = random.uniform(2, 5)

    def update(self, dt):
        self.x += self.vx
        self.y += self.vy
        self.vy += 1.5 * dt
        self.life -= dt
        return self.life > 0

    def draw(self, surf):
        a = max(0, self.life / self.max_life)
        r = int(self.size * a)
        if r > 0:
            c = tuple(min(255, int(v * a)) for v in self.color)
            pygame.draw.circle(surf, c, (int(self.x), int(self.y)), r)


# ╔═══════════════════════════════════════╗
# ║             主渲染器                  ║
# ╚═══════════════════════════════════════╝

class Renderer:
    def __init__(self, screen, clock, fonts):
        self.screen = screen
        self.clock = clock
        self.fonts = fonts
        self.particles = []
        self.pulse = 0.0
        px = ARENA_W + 10
        cw = PANEL_W - 20
        self.chart_winrate = MiniChart(px, 10, cw, 70, "AI WinRate%", C_CHART_WIN)
        self.chart_reward = MiniChart(px, 90, cw, 70, "AI Reward", C_CHART_REW)
        self.chart_eps = MiniChart(px, 170, cw, 70, "Epsilon", C_CHART_EPS)

    def add_explosion_particles(self, x, y):
        for _ in range(12):
            self.particles.append(Particle(
                x * CELL + CELL // 2, y * CELL + CELL // 2,
                random.choice(C_EXPLODE), (2, 6)))

    def add_hit_particles(self, x, y, color):
        for _ in range(8):
            self.particles.append(Particle(
                x * CELL + CELL // 2, y * CELL + CELL // 2,
                color, (1, 3)))

    def draw_arena(self, world):
        self.pulse = (self.pulse + 0.1) % (2 * math.pi)
        dt = 1.0 / max(BASE_FPS, 15)

        # 背景
        self.screen.fill(C_BG)

        # 危险热力图
        dmap = world.get_danger_map()

        # 网格
        for y in range(ARENA_ROWS):
            for x in range(ARENA_COLS):
                rx, ry = x * CELL, y * CELL
                cell = world.grid[y][x]
                if cell == WALL:
                    pygame.draw.rect(self.screen, C_WALL, (rx, ry, CELL, CELL))
                    pygame.draw.rect(self.screen, (80, 85, 100), (rx + 1, ry + 1, CELL - 2, CELL - 2), 1)
                elif cell == BRICK:
                    pygame.draw.rect(self.screen, C_BRICK, (rx, ry, CELL, CELL))
                    pygame.draw.rect(self.screen, C_BRICK_D,
                                     (rx + 2, ry + 2, CELL - 4, CELL - 4))  #砖纹
                    pygame.draw.line(self.screen, C_BRICK_D, (rx, ry + CELL // 2), (rx + CELL, ry + CELL // 2), 1)
                    pygame.draw.line(self.screen, C_BRICK_D,
                                     (rx + CELL // 2, ry), (rx + CELL // 2, ry + CELL), 1)
                else:
                    pygame.draw.rect(self.screen, C_FLOOR, (rx, ry, CELL, CELL))
                    # 危险着色
                    d = dmap[y][x]
                    if d > 0.01:
                        ds = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                        ds.fill((255, 50, 0, int(d * 100)))
                        self.screen.blit(ds, (rx, ry))
                    pygame.draw.rect(self.screen, C_GRID, (rx, ry, CELL, CELL), 1)

        # 道具
        for pu in world.powerups:
            px_, py_ = pu.x * CELL + CELL // 2, pu.y * CELL + CELL // 2
            r = int(CELL * 0.3 + math.sin(self.pulse * 2) * 2)
            colors = [C_POWERUP, C_WARN, C_SHIELD]
            icons = ["R", "S", "B"]
            c = colors[pu.kind % 3]
            pygame.draw.circle(self.screen, c, (px_, py_), r)
            txt = self.fonts["sm"].render(icons[pu.kind % 3], True, (0, 0, 0))
            self.screen.blit(txt, (px_ - txt.get_width() // 2,
                                   py_ - txt.get_height() // 2))

        # 炸弹
        for b in world.bombs:
            bx, by = b.x * CELL + CELL // 2, b.y * CELL + CELL // 2
            urgency = 1.0 - b.timer / BOMB_TIMER
            r = int(CELL * 0.35 + urgency * 4)
            #闪烁
            flash = (b.timer % 2 == 0) if b.timer <= 3 else False
            bc = C_BOMB_FUSE if flash else C_BOMB
            pygame.draw.circle(self.screen, bc, (bx, by), r)
            # 引信
            pygame.draw.line(self.screen, C_BOMB_FUSE,
                             (bx, by - r), (bx + 3, by - r - 5), 2)
            # 计时器
            txt = self.fonts["sm"].render(str(b.timer), True, (0, 0, 0))
            self.screen.blit(txt, (bx - txt.get_width() // 2,
                                   by - txt.get_height() // 2))

        # 爆炸
        for exp in world.explosions:
            ex, ey = exp.x * CELL, exp.y * CELL
            ci = min(exp.timer, len(C_EXPLODE) - 1)
            c = C_EXPLODE[ci]
            pygame.draw.rect(self.screen, c, (ex + 2, ey + 2, CELL - 4, CELL - 4))

        # 玩家
        self._draw_fighter(world.player, C_PLAYER, C_PLAYER_D, "P")
        # AI
        self._draw_fighter(world.ai, C_AI, C_AI_D, "AI")

        # 粒子
        i = 0
        while i < len(self.particles):
            if self.particles[i].update(dt):
                self.particles[i].draw(self.screen)
                i += 1
            else:
                self.particles.pop(i)

        # 竞技场边框
        pygame.draw.rect(self.screen, (80, 80, 120),
                         (0, 0, ARENA_W, ARENA_H), 3)

    def _draw_fighter(self, fighter, color, dark_color, label):
        if fighter.hp <= 0:
            return
        fx = fighter.x * CELL + CELL // 2
        fy = fighter.y * CELL + CELL // 2
        r = CELL // 2 - 3

        # 无敌闪烁
        if fighter.invincible > 0 and fighter.invincible % 2 == 0:
            color = (255, 255, 255)

        # 身体
        pygame.draw.circle(self.screen, dark_color, (fx, fy), r)
        pygame.draw.circle(self.screen, color, (fx, fy), r - 2)

        # 标签
        txt = self.fonts["sm"].render(label, True, (0, 0, 0))
        self.screen.blit(txt, (fx - txt.get_width() // 2,
                                fy - txt.get_height() // 2))

        # HP条
        hp_w = CELL - 6
        hp_h = 4
        hp_x = fighter.x * CELL + 3
        hp_y = fighter.y * CELL - 6
        pygame.draw.rect(self.screen, (40, 40, 40), (hp_x, hp_y, hp_w, hp_h))
        fill = int(hp_w * fighter.hp / fighter.max_hp)
        hc = C_GOOD if fighter.hp > 1 else C_BAD
        pygame.draw.rect(self.screen, hc, (hp_x, hp_y, fill, hp_h))

    def draw_panel(self, world, episode, epsilon, loss,
                   mode, speed, ai_wins, player_wins,
                   total_rounds, gate_val, strategy_gen):
        px = ARENA_W
        pygame.draw.rect(self.screen, C_PANEL,
                         (px, 0, PANEL_W, ARENA_H))
        pygame.draw.line(self.screen, (50, 50, 80),
                         (px, 0), (px, ARENA_H), 2)

        self.chart_winrate.draw(self.screen, self.fonts["sm"])
        self.chart_reward.draw(self.screen, self.fonts["sm"])
        self.chart_eps.draw(self.screen, self.fonts["sm"])

        iy = 250
        ipx = px + 10
        infos = [
            ("Mode", mode, C_HIGHLIGHT if mode == "PvAI" else C_GOOD),
            ("Round", f"{episode}", C_TEXT),
            ("Step", f"{world.step_count}/{MAX_ROUND_STEPS}", C_TEXT),
            ("Player HP", f"{'❤' * world.player.hp}{'♡' * (MAX_HP - world.player.hp)}", C_PLAYER),
            ("AI HP", f"{'❤' * world.ai.hp}{'♡' * (MAX_HP - world.ai.hp)}", C_AI),
            ("", "", C_TEXT),
            ("P Wins", f"{player_wins}", C_PLAYER),
            ("AI Wins", f"{ai_wins}", C_AI),
            ("Total", f"{total_rounds}", C_DIM),
            ("", "", C_TEXT),
            ("Epsilon", f"{epsilon:.4f}", C_TEXT),
            ("Loss", f"{loss:.5f}", C_TEXT),
            ("Gate", f"{gate_val:.2f}", C_WARN),
            ("Gen", f"{strategy_gen}", C_GOOD),
            ("Speed", f"x{speed}", C_TEXT),
        ]
        for label, val, color in infos:
            if label:
                self.screen.blit(self.fonts["sm"].render(f"{label}:", True, C_DIM),
                                 (ipx, iy))
                self.screen.blit(self.fonts["sm"].render(str(val), True, color),
                                 (ipx + 70, iy))
            iy += 15

    def draw_bottom(self, mode):
        by = ARENA_H
        pygame.draw.rect(self.screen, C_BOTTOM, (0, by, WIN_W, BOTTOM_H))
        pygame.draw.line(self.screen, (50, 50, 80), (0, by), (WIN_W, by), 2)

        y1 = by + 8
        y2 = by + 26
        y3 = by + 44
        y4 = by + 60

        self.screen.blit(self.fonts["lg"].render(
            f"Grid Duel Arena v{VERSION}", True, C_HIGHLIGHT), (10, y1))

        self.screen.blit(self.fonts["sm"].render(
            "[Space] Pause  [↑↓] Speed  [1] PvAI  [2] Self-Play  [3] Train",
            True, C_DIM), (10, y2))
        self.screen.blit(self.fonts["sm"].render(
            "[S] Save  [Tab] Help  [WASD] Move  [J] Bomb  [K] Stay",
            True, C_DIM), (10, y3))
        self.screen.blit(self.fonts["sm"].render(
            f"Mode: {mode}  |  Device: {DEVICE}  |  "
            f"Architecture: ReflexNet (Fast+Slow Dual Channel)",
            True, C_DIM), (10, y4))

    def draw_round_result(self, winner, p_wins, ai_wins, total):
        ov = pygame.Surface((ARENA_W, ARENA_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 160))
        self.screen.blit(ov, (0, 0))

        cx = ARENA_W // 2
        cy = ARENA_H // 2

        if winner == 'player':
            txt = "PLAYER WINS!"
            color = C_PLAYER
        elif winner == 'ai':
            txt = "AI WINS!"
            color = C_AI
        else:
            txt = "DRAW!"
            color = C_HIGHLIGHT

        title = self.fonts["xl"].render(txt, True, color)
        self.screen.blit(title, (cx - title.get_width() // 2, cy - 50))

        score = self.fonts["md"].render(
            f"Player {p_wins}  :  {ai_wins} AI  (of {total})",
            True, C_TEXT)
        self.screen.blit(score, (cx - score.get_width() // 2, cy + 10))

        hint = self.fonts["sm"].render(
            "Press [N] Next Round  |  [Esc] Quit", True, C_DIM)
        self.screen.blit(hint, (cx - hint.get_width() // 2, cy + 40))


# ╔═══════════════════════════════════════╗
# ║             存档/加载                 ║
# ╚═══════════════════════════════════════╝

def save_checkpoint(net, target_net, optimizer, memory, stats,
                    pool, episode, best_reward):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({
        "version": VERSION,
        "episode": episode,
        "best_reward": best_reward,
        "net": net.state_dict(),
        "target": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, CKPT_MODEL)

    # 统计信息
    with open(CKPT_STATS, "w") as f:
        json.dump(stats, f, indent=2)

    pool.save(CKPT_POOL)
    print(f"  💾 Saved ep={episode} best_r={best_reward:.1f}")


def save_best(net, best_reward, episode):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({
        "version": VERSION,
        "episode": episode,
        "best_reward": best_reward,
        "net": net.state_dict(),
    }, CKPT_BEST)
    print(f"  🏆 Best model R={best_reward:.1f}")


def load_checkpoint(net, target_net, optimizer, pool):
    if not os.path.exists(CKPT_MODEL):
        print("  🆕 No checkpoint, starting fresh")
        return 0, -1e9, {"ai_wins": 0, "player_wins": 0, "draws": 0,
                          "rewards": [], "winrates": [], "losses": []}

    ckpt = torch.load(CKPT_MODEL, map_location=DEVICE, weights_only=False)
    net.load_state_dict(ckpt["net"])
    target_net.load_state_dict(ckpt.get("target", ckpt["net"]))
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception:
        print("  ⚠ Optimizer re-init")

    pool.load(CKPT_POOL)

    stats = {"ai_wins": 0, "player_wins": 0, "draws": 0,
             "rewards": [], "winrates": [], "losses": []}
    if os.path.exists(CKPT_STATS):
        try:
            with open(CKPT_STATS) as f:
                stats = json.load(f)
        except Exception:
            pass

    ep = ckpt.get("episode", 0)
    best = ckpt.get("best_reward", -1e9)
    print(f"  ✅ Loaded v{ckpt.get('version', '?')} ep={ep} best={best:.1f}")
    return ep, best, stats


# ╔═══════════════════════════════════════╗
# ║        规则型AI（早期对手）           ║
# ╚═══════════════════════════════════════╝

def rule_based_ai(world):
    """
    简单规则AI，作为早期训练对手和备选
    策略：逃离危险 > 放炸弹 > 靠近玩家
    """
    me = world.ai
    enemy = world.player

    # 1. 检查是否在危险区
    in_danger = False
    for b in world.bombs:
        if (b.x == me.x and abs(b.y - me.y) <= b.power) or \
           (b.y == me.y and abs(b.x - me.x) <= b.power):
            if b.timer <= 4:
                in_danger = True
                break

    if in_danger:
        #逃跑：找安全方向
        best_dir = 5  # 默认不动
        best_safety = -1
        for act in range(4):
            dx, dy = DIR_MAP[act]
            nx, ny = me.x + dx, me.y + dy
            if not world._can_move(nx, ny):
                continue
            if nx == enemy.x and ny == enemy.y:
                continue
            # 评估安全性
            safe = True
            for b in world.bombs:
                if (b.x == nx and abs(b.y - ny) <= b.power) or \
                   (b.y == ny and abs(b.x - nx) <= b.power):
                    safe = False
            safety = 2 if safe else 0
            # 远离炸弹加分
            for b in world.bombs:
                safety += abs(b.x - nx) + abs(b.y - ny)
            if safety > best_safety:
                best_safety = safety
                best_dir = act
        return best_dir

    # 2. 如果靠近玩家且能放炸弹
    dist = abs(me.x - enemy.x) + abs(me.y - enemy.y)
    if dist <= 2 and me.bomb_cooldown <= 0 and me.active_bombs < me.max_bombs:
        # 确保放炸弹后有地方跑
        can_escape = False
        for act in range(4):
            dx, dy = DIR_MAP[act]
            nx, ny = me.x + dx, me.y + dy
            if world._can_move(nx, ny) and not (nx == enemy.x and ny == enemy.y):
                can_escape = True
                break
        if can_escape:
            return 4  # 放炸弹

    # 3. 向玩家移动
    best_dir = 5
    best_dist = dist
    for act in range(4):
        dx, dy = DIR_MAP[act]
        nx, ny = me.x + dx, me.y + dy
        if not world._can_move(nx, ny):
            continue
        if nx == enemy.x and ny == enemy.y:
            continue
        nd = abs(nx - enemy.x) + abs(ny - enemy.y)
        if nd < best_dist:
            best_dist = nd
            best_dir = act

    return best_dir


# ╔═══════════════════════════════════════╗
# ║             主程序                    ║
# ╚═══════════════════════════════════════╝

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"Grid Duel Arena v{VERSION}")
    clock = pygame.time.Clock()

    fonts = {
        "sm": pygame.font.SysFont("consolas", 11),
        "md": pygame.font.SysFont("consolas", 13, bold=True),
        "lg": pygame.font.SysFont("consolas", 17, bold=True),
        "xl": pygame.font.SysFont("consolas", 30, bold=True),
    }

    # 网络
    net = ReflexNet().to(DEVICE)
    target_net = ReflexNet().to(DEVICE)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    memory = LightPER(MEMORY_SIZE)
    nstep = NStepBuffer(N_STEP, GAMMA)
    pool = StrategyPool(max_size=8)

    # 对手网络（用于self-play）
    opponent_net = ReflexNet().to(DEVICE)
    opponent_net.eval()

    # 加载
    start_ep, best_reward, stats = load_checkpoint(net, target_net, optimizer, pool)
    renderer = Renderer(screen, clock, fonts)

    # 恢复图表
    for v in stats.get("winrates", [])[-150:]:
        renderer.chart_winrate.add(v)
    for v in stats.get("rewards", [])[-150:]:
        renderer.chart_reward.add(v)
    for v in stats.get("losses", [])[-150:]:
        renderer.chart_eps.add(v)

    # 状态
    mode = "PvAI"     # PvAI, SelfPlay, Train
    speed = 1
    paused = False
    global_step = start_ep * 100
    last_loss = 0.0
    gate_val = 0.5
    player_wins = stats.get("player_wins", 0)
    ai_wins = stats.get("ai_wins", 0)
    total_rounds = player_wins + ai_wins + stats.get("draws", 0)
    player_action = 5

    print(f"\n  🏟️ Grid Duel Arena v{VERSION} | {DEVICE}")
    print(f"     [1] Player vs AI  [2] Self-Play  [3] Fast Train\n")

    episode = start_ep

    while True:
        world = DuelArena(seed=random.randint(0, 2**31))
        obs = world.reset()
        total_reward = 0.0
        nstep.reset()
        round_over = False
        show_result = False
        result_timer = 0

        epsilon = max(EPS_END, EPS_START - global_step / EPS_DECAY)

        # 选择对手策略
        use_pool_opponent = (mode in ("SelfPlay", "Train") and pool.pool and random.random() < 0.3)
        if use_pool_opponent:
            opp = pool.sample_opponent()
            opponent_net.load_state_dict(opp[1])
            opp_name = opp[0]
        else:
            opp_name = "Rule"

        while not round_over:
            # === 事件处理 ===
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    stats["player_wins"] = player_wins
                    stats["ai_wins"] = ai_wins
                    save_checkpoint(net, target_net, optimizer, memory,
                                    stats, pool, episode, best_reward)
                    pygame.quit()
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        stats["player_wins"] = player_wins
                        stats["ai_wins"] = ai_wins
                        save_checkpoint(net, target_net, optimizer, memory,
                                        stats, pool, episode, best_reward)
                        pygame.quit()
                        return
                    elif ev.key == pygame.K_SPACE:
                        paused = not paused
                    elif ev.key == pygame.K_UP:
                        speed = min(speed + 1, 30)
                    elif ev.key == pygame.K_DOWN:
                        speed = max(speed - 1, 1)
                    elif ev.key == pygame.K_1:
                        mode = "PvAI"
                        speed = 1
                    elif ev.key == pygame.K_2:
                        mode = "SelfPlay"
                        speed = 3
                    elif ev.key == pygame.K_3:
                        mode = "Train"
                        speed = 15
                    elif ev.key == pygame.K_s:
                        stats["player_wins"] = player_wins
                        stats["ai_wins"] = ai_wins
                        save_checkpoint(net, target_net, optimizer, memory,
                                        stats, pool, episode, best_reward)
                    elif ev.key == pygame.K_n and show_result:
                        round_over = True
                        continue    # 玩家控制
                    if mode == "PvAI":
                        if ev.key in (pygame.K_w, pygame.K_UP):
                            player_action = 0
                        elif ev.key in (pygame.K_s, pygame.K_DOWN):
                            player_action = 1
                        elif ev.key == pygame.K_a:
                            player_action = 2
                        elif ev.key == pygame.K_d:
                            player_action = 3
                        elif ev.key == pygame.K_j:
                            player_action = 4
                        elif ev.key == pygame.K_k:
                            player_action = 5
            # 修复WASD映射
            keys = pygame.key.get_pressed()
            if mode == "PvAI":
                if keys[pygame.K_w]:
                    player_action = 0
                elif keys[pygame.K_s] and not keys[pygame.K_LCTRL]:
                    player_action = 1
                elif keys[pygame.K_a]:
                    player_action = 2
                elif keys[pygame.K_d]:
                    player_action = 3

            if paused and not show_result:
                renderer.draw_arena(world)
                renderer.draw_panel(world, episode, epsilon, last_loss,
                                    mode, speed, ai_wins, player_wins,
                                    total_rounds, gate_val, pool.generation)
                renderer.draw_bottom(mode)

                #暂停文字
                ptxt = fonts["lg"].render("PAUSED", True, C_WARN)
                screen.blit(ptxt, (ARENA_W // 2 - ptxt.get_width() // 2, ARENA_H // 2 - ptxt.get_height() // 2))
                pygame.display.flip()
                clock.tick(15)
                continue

            if show_result:
                renderer.draw_arena(world)
                renderer.draw_panel(world, episode, epsilon, last_loss,
                                    mode, speed, ai_wins, player_wins,
                                    total_rounds, gate_val, pool.generation)
                renderer.draw_bottom(mode)
                renderer.draw_round_result(world.winner, player_wins, ai_wins, total_rounds)
                pygame.display.flip()
                clock.tick(15)
                if mode in ("SelfPlay", "Train"):
                    result_timer += 1
                    if result_timer > (3 if mode == "Train" else 15):
                        round_over = True
                continue

            # === 决定玩家动作 ===
            if mode == "PvAI":
                p_act = player_action
                player_action = 5  # 重置
            elif use_pool_opponent:
                # 历史策略作为玩家
                st = world.get_state(for_ai=False)
                st_t = torch.tensor(st, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q = opponent_net(st_t)
                p_act = q.argmax(dim=-1).item()
            else:
                # 规则AI作为玩家
                p_act = rule_based_ai_for_player(world)

            # === AI决策 ===
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = net(obs_t)
                gate_val = net.get_gate_value(obs_t)

            if random.random() < epsilon:
                ai_act = random.randint(0, ACTION_DIM - 1)
            else:
                ai_act = q_values.argmax(dim=-1).item()

            # === 执行 ===
            prev_ai_hp = world.ai.hp
            prev_player_hp = world.player.hp
            obs2, reward, done = world.step(p_act, ai_act)
            total_reward += reward
            global_step += 1

            #粒子效果
            for exp in world.explosions:
                if exp.timer == 4:  # 刚爆炸
                    renderer.add_explosion_particles(exp.x, exp.y)
            if world.ai.hp < prev_ai_hp:
                renderer.add_hit_particles(world.ai.x, world.ai.y, C_AI)
            if world.player.hp < prev_player_hp:
                renderer.add_hit_particles(
                    world.player.x, world.player.y, C_PLAYER)

            # 存储经验
            nstep.push((obs, ai_act, reward, obs2, float(done)))
            nt = nstep.get()
            if nt:
                memory.push(nt)
            obs = obs2

            # === 训练 ===
            if len(memory) >= BATCH_SIZE:
                batch, idx, isw = memory.sample(BATCH_SIZE)
                bs = torch.tensor(np.array([t[0] for t in batch]),
                                  dtype=torch.float32)
                ba = torch.tensor([t[1] for t in batch],
                                  dtype=torch.long).unsqueeze(-1)
                br = torch.tensor([t[2] for t in batch],
                                  dtype=torch.float32).unsqueeze(-1)
                bs2 = torch.tensor(np.array([t[3] for t in batch]),
                                   dtype=torch.float32)
                bd = torch.tensor([t[4] for t in batch],
                                  dtype=torch.float32).unsqueeze(-1)

                # Double DQN
                with torch.no_grad():
                    best_a = net(bs2).argmax(dim=-1, keepdim=True)
                    q_next = target_net(bs2).gather(1, best_a)
                    target = br + GAMMA ** N_STEP * q_next * (1 - bd)

                q_current = net(bs).gather(1, ba)
                td_error = (target - q_current).detach().squeeze().numpy()
                loss = (isw.unsqueeze(-1) * (q_current - target) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                optimizer.step()

                memory.update_priorities(idx, td_error)
                last_loss = loss.item()
                # 软更新
                for tp, sp in zip(target_net.parameters(), net.parameters()):
                    tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)

            # === 渲染 ===
            if mode != "Train" or global_step % 3 == 0:
                renderer.draw_arena(world)
                renderer.draw_panel(world, episode, epsilon, last_loss,
                                    mode, speed, ai_wins, player_wins,
                                    total_rounds, gate_val, pool.generation)
                renderer.draw_bottom(mode)
                pygame.display.flip()
            fps = BASE_FPS * speed
            if mode == "Train":
                fps = max(fps, 120)
            clock.tick(fps)

            if done:
                for t in nstep.flush():
                    memory.push(t)
                show_result = True
                result_timer = 0
                total_rounds += 1
                if world.winner == 'ai':
                    ai_wins += 1
                elif world.winner == 'player':
                    player_wins += 1

                # 更新统计
                wr = ai_wins / max(total_rounds, 1) * 100
                stats.setdefault("rewards", []).append(total_reward)
                stats.setdefault("winrates", []).append(wr)
                stats.setdefault("losses", []).append(last_loss)

                renderer.chart_winrate.add(wr)
                renderer.chart_reward.add(total_reward)
                renderer.chart_eps.add(epsilon)

                # 打印
                print(f"EP {episode:5d}│"
                      f"{'AI WIN' if world.winner == 'ai' else 'P WIN' if world.winner == 'player' else 'DRAW':>6s}│"
                      f"R:{total_reward:7.1f}│"
                      f"AI:{ai_wins} P:{player_wins}│"
                      f"WR:{wr:5.1f}%│"
                      f"ε:{epsilon:.3f}│"
                      f"Gate:{gate_val:.2f}│"
                      f"Opp:{opp_name}│"
                      f"Mem:{len(memory)}")

                # 策略池更新（每20轮克隆一次）
                if episode % 20 == 0 and episode > 0:
                    fitness = sum(stats["rewards"][-20:]) / 20
                    pool.add(f"gen{pool.generation}_ep{episode}",
                             net.state_dict(), fitness)

                if total_reward > best_reward:
                    best_reward = total_reward
                    save_best(net, best_reward, episode)

                if (episode + 1) % 50 == 0:
                    stats["player_wins"] = player_wins
                    stats["ai_wins"] = ai_wins
                    save_checkpoint(net, target_net, optimizer, memory,
                                    stats, pool, episode + 1, best_reward)

                episode += 1

        # round结束，继续下一轮


def rule_based_ai_for_player(world):
    """规则AI，但控制player角色（用于self-play/train模式）"""
    me = world.player
    enemy = world.ai

    # 检查危险
    in_danger = False
    for b in world.bombs:
        if (b.x == me.x and abs(b.y - me.y) <= b.power) or \
           (b.y == me.y and abs(b.x - me.x) <= b.power):
            if b.timer <= 4:
                in_danger = True
                break

    if in_danger:
        best_dir = 5
        best_safety = -1
        for act in range(4):
            dx, dy = DIR_MAP[act]
            nx, ny = me.x + dx, me.y + dy
            if not world._can_move(nx, ny):
                continue
            if nx == enemy.x and ny == enemy.y:
                continue
            safe = True
            safety = 0
            for b in world.bombs:
                if (b.x == nx and abs(b.y - ny) <= b.power) or \
                   (b.y == ny and abs(b.x - nx) <= b.power):
                    safe = False
                safety += abs(b.x - nx) + abs(b.y - ny)
            safety += (10 if safe else 0)
            if safety > best_safety:
                best_safety = safety
                best_dir = act
        return best_dir

    dist = abs(me.x - enemy.x) + abs(me.y - enemy.y)
    if dist <= 3 and me.bomb_cooldown <= 0 and me.active_bombs < me.max_bombs:
        can_escape = False
        for act in range(4):
            dx, dy = DIR_MAP[act]
            nx, ny = me.x + dx, me.y + dy
            if world._can_move(nx, ny) and not (nx == enemy.x and ny == enemy.y):
                can_escape = True
                break
        if can_escape and random.random() < 0.6:
            return 4

    # 随机性让训练更鲁棒
    if random.random() < 0.15:
        return random.randint(0, 5)

    best_dir = 5
    best_dist = dist
    for act in range(4):
        dx, dy = DIR_MAP[act]
        nx, ny = me.x + dx, me.y + dy
        if not world._can_move(nx, ny):
            continue
        if nx == enemy.x and ny == enemy.y:
            continue
        nd = abs(nx - enemy.x) + abs(ny - enemy.y)
        if nd < best_dist:
            best_dist = nd
            best_dir = act
    return best_dir


if __name__ == "__main__":
    main()