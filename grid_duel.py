#!/usr/bin/env python3
"""
Grid Duel Arena v3.2 — Complete Fixed Version
==============================================
v3.2 修复:
  ✦ spawn子进程不再重跑全局初始化 (消除16次重复打印)
  ✦ Worker端攒整局数据批量发送 (Queue序列化量降200倍)
  ✦ 主进程批量接收 + 训练管线优化
  ✦ GUI模式完整保留，兼容v3.0/v3.1存档

pip install torch numpy
pip install pygame  # 仅GUI模式需要
"""

import os
import sys
import time
import json
import copy
import math
import random
import pickle
import signal
import argparse
import warnings
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

# ╔════════════════════════════════════════════╗
# ║     全局常量 (纯数值，子进程安全)          ║
# ╚════════════════════════════════════════════╝

VERSION = "3.2"

ARENA_COLS = 13
ARENA_ROWS = 11
CELL = 40
ARENA_W = ARENA_COLS * CELL
ARENA_H = ARENA_ROWS * CELL
PANEL_W = 260
BOTTOM_H = 80
WIN_W = ARENA_W + PANEL_W
WIN_H = ARENA_H + BOTTOM_H

STATE_DIM = 42
ACTION_DIM = 6

GAMMA = 0.97
LR = 5e-4
LR_MIN = 1e-5
TAU = 0.005
EPS_START = 1.0
EPS_END = 0.03
EPS_DECAY = 6000
N_STEP = 3
EVAL_INTERVAL = 25

MAX_HP = 3
BOMB_TIMER = 8
BOMB_RANGE = 2
BOMB_COOLDOWN = 12
POWERUP_INTERVAL = 60
MAX_ROUND_STEPS = 500
BASE_FPS = 8

CKPT_DIR = "grid_duel_ckpt"
CKPT_MODEL = os.path.join(CKPT_DIR, "model.pth")
CKPT_BEST = os.path.join(CKPT_DIR, "best.pth")
CKPT_POOL = os.path.join(CKPT_DIR, "strategy_pool.pkl")
CKPT_STATS = os.path.join(CKPT_DIR, "stats.json")

DIR_MAP = {
    0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0),
    4: (0, 0), 5: (0, 0),
}
DIR_NAMES = ["Up", "Dn", "Lt", "Rt", "Bomb", "Stay"]

EMPTY = 0
WALL = 1
BRICK = 2

#颜色
C_BG = (12, 12, 22)
C_GRID = (25, 25, 40)
C_WALL = (60, 65, 80)
C_WALL_L = (80, 85, 100)
C_BRICK = (140, 100, 60)
C_BRICK_D = (110, 75, 40)
C_FLOOR = (18, 18, 30)
C_PANEL = (16, 16, 28)
C_BOTTOM = (22, 22, 38)
C_TEXT = (180, 180, 210)
C_DIM = (100, 100, 130)
C_GOOD = (60, 255, 140)
C_BAD = (255, 70, 70)
C_WARN = (255, 180, 40)
C_HIGHLIGHT = (255, 210, 50)
C_PLAYER = (0, 200, 255)
C_PLAYER_D = (0, 140, 200)
C_AI = (255, 80, 80)
C_AI_D = (200, 50, 50)
C_BOMB = (255, 220, 50)
C_BOMB_FUSE = (255, 100, 30)
C_EXPLODE = [(255, 255, 200), (255, 200, 50), (255, 120, 20), (200, 50, 0)]
C_POWERUP = (100, 255, 200)
C_SHIELD = (80, 160, 255)
C_CHART_WIN = (80, 200, 255)
C_CHART_REW = (255, 175, 55)
C_CHART_EPS = (200, 100, 255)
C_CHART_LOSS = (255, 100, 100)


# ╔════════════════════════════════════════════╗
# ║             地图生成器                     ║
# ╚════════════════════════════════════════════╝

def generate_arena(seed=None):
    rng = random.Random(seed)
    grid = [[EMPTY] * ARENA_COLS for _ in range(ARENA_ROWS)]
    for x in range(ARENA_COLS):
        grid[0][x] = WALL
        grid[ARENA_ROWS - 1][x] = WALL
    for y in range(ARENA_ROWS):
        grid[y][0] = WALL
        grid[y][ARENA_COLS - 1] = WALL
    for y in range(2, ARENA_ROWS - 2, 2):
        for x in range(2, ARENA_COLS - 2, 2):
            grid[y][x] = WALL
    half_x = ARENA_COLS // 2
    for y in range(1, ARENA_ROWS - 1):
        for x in range(1, half_x + 1):
            if grid[y][x] != EMPTY:
                continue
            mx = ARENA_COLS - 1 - x
            if grid[y][mx] != EMPTY:
                continue
            if y <= 2 and x <= 2:
                continue
            if y <= 2 and mx >= ARENA_COLS - 3:
                continue
            if y >= ARENA_ROWS - 3 and x <= 2:
                continue
            if y >= ARENA_ROWS - 3 and mx >= ARENA_COLS - 3:
                continue
            if rng.random() < 0.35:
                grid[y][x] = BRICK
                grid[y][mx] = BRICK
    return grid


# ╔════════════════════════════════════════════╗
# ║             游戏对象定义                   ║
# ╚════════════════════════════════════════════╝

class Bomb:
    __slots__ = ["x", "y", "timer", "owner", "power"]
    def __init__(self, x, y, owner, power=BOMB_RANGE):
        self.x, self.y, self.timer, self.owner, self.power = x, y, BOMB_TIMER, owner, power


class Explosion:
    __slots__ = ["x", "y", "timer"]
    def __init__(self, x, y):
        self.x, self.y, self.timer = x, y, 4


class PowerUp:
    __slots__ = ["x", "y", "kind"]
    def __init__(self, x, y, kind):
        self.x, self.y, self.kind = x, y, kind


class Fighter:
    def __init__(self, x, y, is_ai=False):
        self.x, self.y = x, y
        self.hp = MAX_HP
        self.max_hp = MAX_HP
        self.bomb_cooldown = 0
        self.bomb_power = BOMB_RANGE
        self.max_bombs = 1
        self.active_bombs = 0
        self.speed = 1
        self.invincible = 0
        self.is_ai = is_ai
        self.score = 0
        self.kills = 0
        self.deaths = 0
        self.last_action = 5
        self.prev_hp = MAX_HP

    def reset(self, x, y):
        self.x, self.y = x, y
        self.hp = self.max_hp
        self.prev_hp = self.max_hp
        self.bomb_cooldown = 0
        self.active_bombs = 0
        self.invincible = 0
        self.last_action = 5


# ╔════════════════════════════════════════════╗
# ║             DuelArena 游戏世界             ║
# ╚════════════════════════════════════════════╝

class DuelArena:
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
        self.winner = None
        self.powerup_timer = POWERUP_INTERVAL
        self.rng = random.Random(seed)
        self.new_explosions = []
        self.last_ai_move_valid = True
        self.bricks_broken_by_ai = 0
        self.reset()

    def reset(self):
        self.grid = generate_arena(self.seed)
        self.player.reset(1, ARENA_ROWS - 2)
        self.ai.reset(ARENA_COLS - 2, 1)
        self.bombs.clear()
        self.explosions.clear()
        self.powerups.clear()
        self.new_explosions.clear()
        self.step_count = 0
        self.round_over = False
        self.winner = None
        self.powerup_timer = POWERUP_INTERVAL
        self.last_ai_move_valid = True
        self.bricks_broken_by_ai = 0
        return self.get_state(for_ai=True)

    def _can_move(self, x, y):
        if x < 0 or x >= ARENA_COLS or y < 0 or y >= ARENA_ROWS:
            return False
        if self.grid[y][x] in (WALL, BRICK):
            return False
        for b in self.bombs:
            if b.x == x and b.y == y:
                return False
        return True

    def _do_move(self, fighter, action):
        fighter.last_action = action
        fighter.prev_hp = fighter.hp
        if fighter.bomb_cooldown > 0:
            fighter.bomb_cooldown -= 1
        if fighter.invincible > 0:
            fighter.invincible -= 1
        move_valid = True
        if action == 4:
            if (fighter.bomb_cooldown <= 0 and fighter.active_bombs < fighter.max_bombs):
                has_bomb = any(
                    b.x == fighter.x and b.y == fighter.y
                    for b in self.bombs)
                if not has_bomb:
                    self.bombs.append(
                        Bomb(fighter.x, fighter.y, fighter, fighter.bomb_power))
                    fighter.active_bombs += 1
                    fighter.bomb_cooldown = BOMB_COOLDOWN
                else:
                    move_valid = False
            else:
                move_valid = False
        elif action < 4:
            dx, dy = DIR_MAP[action]
            nx, ny = fighter.x + dx, fighter.y + dy
            if self._can_move(nx, ny):
                other = self.ai if not fighter.is_ai else self.player
                if not (nx == other.x and ny == other.y):
                    fighter.x, fighter.y = nx, ny
                else:
                    move_valid = False
            else:
                move_valid = False
        if fighter.is_ai:
            self.last_ai_move_valid = move_valid

    def _explode_bomb(self, bomb):
        cells = [(bomb.x, bomb.y)]
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            for r in range(1, bomb.power + 1):
                ex, ey = bomb.x + dx * r, bomb.y + dy * r
                if ex < 0 or ex >= ARENA_COLS or ey < 0 or ey >= ARENA_ROWS:
                    break
                if self.grid[ey][ex] == WALL:
                    break
                cells.append((ex, ey))
                if self.grid[ey][ex] == BRICK:
                    self.grid[ey][ex] = EMPTY
                    if bomb.owner is self.ai:
                        self.bricks_broken_by_ai += 1
                    if self.rng.random() < 0.25:
                        self.powerups.append(
                            PowerUp(ex, ey, self.rng.randint(0, 2)))
                    break
        return cells

    def _check_explosion_hit(self, cells):
        for cx, cy in cells:
            for fighter in [self.player, self.ai]:
                if (fighter.x == cx and fighter.y == cy
                        and fighter.invincible <= 0 and fighter.hp > 0):
                    fighter.hp -= 1
                    fighter.invincible = 6
                    if fighter.hp <= 0:
                        other = (self.ai if fighter is self.player else self.player)
                        other.kills += 1
                        fighter.deaths += 1

    def _check_powerups(self):
        for fighter in [self.player, self.ai]:
            i = 0
            while i < len(self.powerups):
                pu = self.powerups[i]
                if pu.x == fighter.x and pu.y == fighter.y:
                    if pu.kind == 0:
                        fighter.bomb_power = min(
                            fighter.bomb_power + 1, 5)
                    elif pu.kind == 1:
                        fighter.speed = min(fighter.speed + 1, 3)
                    elif pu.kind == 2:
                        fighter.max_bombs = min(
                            fighter.max_bombs + 1, 3)
                    self.powerups.pop(i)
                else:
                    i += 1

    def step(self, player_action, ai_action):
        if self.round_over:
            return self.get_state(for_ai=True), 0.0, True
        self.step_count += 1
        self.new_explosions.clear()
        self._do_move(self.player, player_action)
        self._do_move(self.ai, ai_action)

        exploded_cells = []
        changed = True
        while changed:
            changed = False
            to_explode = []
            for i, bomb in enumerate(self.bombs):
                bomb.timer -= 1
                if bomb.timer <= 0:
                    to_explode.append(i)
            if to_explode:
                changed = True
                for i in sorted(to_explode, reverse=True):
                    bomb = self.bombs[i]
                    cells = self._explode_bomb(bomb)
                    exploded_cells.extend(cells)
                    for cx, cy in cells:
                        exp = Explosion(cx, cy)
                        self.explosions.append(exp)
                        self.new_explosions.append(exp)
                    for other_bomb in self.bombs:
                        if other_bomb is not bomb and other_bomb.timer > 0:
                            for cx, cy in cells:
                                if (other_bomb.x == cx
                                        and other_bomb.y == cy):
                                    other_bomb.timer = 0
                    bomb.owner.active_bombs = max(
                        0, bomb.owner.active_bombs - 1)
                    self.bombs.pop(i)

        self._check_explosion_hit(exploded_cells)

        i = 0
        while i < len(self.explosions):
            self.explosions[i].timer -= 1
            if self.explosions[i].timer <= 0:
                self.explosions.pop(i)
            else:
                i += 1

        self._check_powerups()
        self.powerup_timer -= 1
        if self.powerup_timer <= 0:
            self.powerup_timer = POWERUP_INTERVAL
            self._spawn_random_powerup()

        ai_reward = 0.0
        done = False
        if self.player.hp <= 0 and self.ai.hp <= 0:
            self.winner = "draw"
            self.round_over = True
            done = True
        elif self.player.hp <= 0:
            self.winner = "ai"
            self.round_over = True
            done = True
            ai_reward = 20.0
        elif self.ai.hp <= 0:
            self.winner = "player"
            self.round_over = True
            done = True
            ai_reward = -20.0
        elif self.step_count >= MAX_ROUND_STEPS:
            if self.ai.hp > self.player.hp:
                self.winner = "ai"
                ai_reward = 10.0
            elif self.player.hp > self.ai.hp:
                self.winner = "player"
                ai_reward = -10.0
            else:
                self.winner = "draw"
                ai_reward = -2.0
            self.round_over = True
            done = True
        if not done:
            ai_reward = self._compute_shaping_reward()
            
        return self.get_state(for_ai=True), ai_reward, done

    def _compute_shaping_reward(self):
        reward = 0.005
        dist = abs(self.ai.x - self.player.x) + abs(
            self.ai.y - self.player.y)
        if 2 <= dist <= 4:
            reward += 0.08
        elif dist <= 1:
            reward -= 0.02
        elif dist >= 9:
            reward -= 0.03
        for exp in self.explosions:
            if self.ai.x == exp.x and self.ai.y == exp.y:
                reward -= 0.5
        for b in self.bombs:
            in_line = ((b.x == self.ai.x
                        and abs(b.y - self.ai.y) <= b.power)
                       or (b.y == self.ai.y
                           and abs(b.x - self.ai.x) <= b.power))
            if in_line:
                urgency = 1.0 - b.timer / BOMB_TIMER
                if b.timer <= 2:
                    reward -= 0.4 * urgency
                elif b.timer <= 4:
                    reward -= 0.15 * urgency
        hp_diff = self.ai.hp - self.player.hp
        reward += hp_diff * 0.04
        if self.player.hp < self.player.prev_hp:
            reward += 1.5
        if self.ai.hp < self.ai.prev_hp:
            reward -= 1.0
        if not self.last_ai_move_valid:
            reward -= 0.08
        if self.ai.last_action == 4 and self.last_ai_move_valid:
            reward += 0.3 if dist <= 4 else 0.05
        cx = abs(self.ai.x - ARENA_COLS // 2)
        cy = abs(self.ai.y - ARENA_ROWS // 2)
        if cx + cy <= 3:
            reward += 0.02
        escape_routes = 0
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            if self._can_move(self.ai.x + dx, self.ai.y + dy):
                escape_routes += 1
        if escape_routes == 0:
            reward -= 0.15
        if self.bricks_broken_by_ai > 0:
            reward += 0.1 * self.bricks_broken_by_ai
            self.bricks_broken_by_ai = 0
        return np.clip(reward, -3.0, 3.0)

    def _spawn_random_powerup(self):
        occupied = {(self.player.x, self.player.y),
                    (self.ai.x, self.ai.y)}
        for b in self.bombs:
            occupied.add((b.x, b.y))
        for pu in self.powerups:
            occupied.add((pu.x, pu.y))
        empties = []
        for y in range(1, ARENA_ROWS - 1):
            for x in range(1, ARENA_COLS - 1):
                if (self.grid[y][x] == EMPTY
                        and (x, y) not in occupied):
                    empties.append((x, y))
        if empties and len(self.powerups) < 4:
            x, y = self.rng.choice(empties)
            self.powerups.append(
                PowerUp(x, y, self.rng.randint(0, 2)))

    def get_state(self, for_ai=True):
        me = self.ai if for_ai else self.player
        enemy = self.player if for_ai else self.ai
        features = []
        features.append(me.x / ARENA_COLS)
        features.append(me.y / ARENA_ROWS)
        features.append(me.hp / me.max_hp)
        features.append(me.bomb_cooldown / BOMB_COOLDOWN)
        features.append(me.active_bombs / max(me.max_bombs, 1))
        features.append(me.invincible / 6.0)
        features.append((enemy.x - me.x) / ARENA_COLS)
        features.append((enemy.y - me.y) / ARENA_ROWS)
        features.append(enemy.hp / enemy.max_hp)
        dist = abs(enemy.x - me.x) + abs(enemy.y - me.y)
        features.append(dist / (ARENA_COLS + ARENA_ROWS))
        angle = math.atan2(enemy.y - me.y, enemy.x - me.x)
        features.append(angle / math.pi)
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in dirs:
            wall_dist = 0
            for r in range(1, max(ARENA_COLS, ARENA_ROWS)):
                cx_, cy_ = me.x + dx * r, me.y + dy * r
                if (cx_ < 0 or cx_ >= ARENA_COLS
                        or cy_ < 0 or cy_ >= ARENA_ROWS):
                    break
                if self.grid[cy_][cx_] in (WALL, BRICK):
                    break
                wall_dist = r
            features.append(wall_dist / max(ARENA_COLS, ARENA_ROWS))
            danger = 0.0
            for b in self.bombs:
                if (dx != 0 and b.y == me.y
                        and 0 < (b.x - me.x) * dx <= b.power + 1):
                    danger = max(danger, 1.0 - b.timer / BOMB_TIMER)
                if (dy != 0 and b.x == me.x
                        and 0 < (b.y - me.y) * dy <= b.power + 1):
                    danger = max(danger, 1.0 - b.timer / BOMB_TIMER)
            features.append(danger)
            has_exp = 0.0
            for exp in self.explosions:
                if (dx != 0 and exp.y == me.y
                        and 0 < (exp.x - me.x) * dx <= 2):
                    has_exp = 1.0
                if (dy != 0 and exp.x == me.x
                        and 0 < (exp.y - me.y) * dy <= 2):
                    has_exp = 1.0
            features.append(has_exp)
        my_bomb_count = sum(1 for b in self.bombs if b.owner is me)
        enemy_bomb_count = sum(
            1 for b in self.bombs if b.owner is enemy)
        features.append(my_bomb_count / 3.0)
        features.append(enemy_bomb_count / 3.0)
        min_bomb_dist = 1.0
        min_bomb_timer = 1.0
        for b in self.bombs:
            bd = ((abs(b.x - me.x) + abs(b.y - me.y)) / (ARENA_COLS + ARENA_ROWS))
            if bd < min_bomb_dist:
                min_bomb_dist = bd
                min_bomb_timer = b.timer / BOMB_TIMER
        features.append(min_bomb_dist)
        features.append(min_bomb_timer)
        in_danger = 0.0
        for b in self.bombs:
            if ((b.x == me.x and abs(b.y - me.y) <= b.power)
                    or (b.y == me.y
                        and abs(b.x - me.x) <= b.power)):
                in_danger = max(
                    in_danger, 1.0 - b.timer / BOMB_TIMER)
        features.append(in_danger)
        stuck = 1.0
        for ddx, ddy in dirs:
            if self._can_move(me.x + ddx, me.y + ddy):
                stuck = 0.0
                break
        features.append(stuck)
        min_pu_dist = 1.0
        min_pu_dx = 0.0
        min_pu_dy = 0.0
        for pu in self.powerups:
            pd = ((abs(pu.x - me.x) + abs(pu.y - me.y))
                  / (ARENA_COLS + ARENA_ROWS))
            if pd < min_pu_dist:
                min_pu_dist = pd
                min_pu_dx = (pu.x - me.x) / ARENA_COLS
                min_pu_dy = (pu.y - me.y) / ARENA_ROWS
        features.append(min_pu_dist)
        features.append(min_pu_dx)
        features.append(min_pu_dy)
        features.append(self.step_count / MAX_ROUND_STEPS)
        features.append(me.bomb_power / 5.0)
        features.append(me.max_bombs / 3.0)
        features.append(len(self.powerups) / 4.0)
        while len(features) < STATE_DIM:
            features.append(0.0)
        return np.array(features[:STATE_DIM], dtype=np.float32)

    def get_danger_map(self):
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
                        dmap[ey][ex] = max(
                            dmap[ey][ex], urgency * 0.7)
                    else:
                        break
        for exp in self.explosions:
            if (0 <= exp.y < ARENA_ROWS
                    and 0 <= exp.x < ARENA_COLS):
                dmap[exp.y][exp.x] = 1.0
        return dmap


# ╔════════════════════════════════════════════╗
# ║      神经网络 (ReflexNet)                  ║
# ╚════════════════════════════════════════════╝

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ReflexNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden=128):
        super().__init__()
        self.hidden = hidden
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fast_net = nn.Sequential(
            nn.Linear(18, 48), Swish(),
            nn.Linear(48, 24), Swish(),
            nn.Linear(24, action_dim),)
        self.slow_net = nn.Sequential(
            nn.Linear(state_dim, hidden), Swish(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), Swish(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2), Swish(),
            nn.Linear(hidden // 2, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(action_dim * 2, 32), Swish(),
            nn.Linear(32, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(action_dim * 2, 32), Swish(),
            nn.Linear(32, action_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(state_dim, 16), nn.Tanh(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        fast_feat = x[:, 11:29] if x.shape[-1] >= 29 else x[:, :18]
        fast_q = self.fast_net(fast_feat)
        slow_q = self.slow_net(x)
        g = self.gate(x)
        combined = torch.cat(
            [fast_q * g, slow_q * (1 - g)], dim=-1)
        v = self.value_head(combined)
        a = self.advantage_head(combined)
        return v + a - a.mean(dim=-1, keepdim=True)

    def get_gate_value(self, x):
        with torch.no_grad():
            return self.gate(x).mean().item()

    def get_q_values(self, state_np, device):
        with torch.no_grad():
            st = torch.tensor(
                state_np, dtype=torch.float32,
                device=device).unsqueeze(0)
            return self(st).squeeze(0).cpu().numpy()


# ╔════════════════════════════════════════════╗
# ║      经验回放 / N-Step /策略池             ║
# ╚════════════════════════════════════════════╝

class LightPER:
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
        s = p.sum()
        if s == 0:
            p = np.ones(n, dtype=np.float64) / n
        else:
            p /= s
        size = min(batch_size, n)
        idx = np.random.choice(n, size, p=p, replace=False)
        samples = [self.buffer[i] for i in idx]
        w = (n * p[idx]) ** (-beta)
        wmax = w.max()
        if wmax > 0:
            w /= wmax
        return samples, idx, torch.tensor(w, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            if 0 <= i < len(self.priorities):
                self.priorities[i] = abs(td) + 1e-6

    def __len__(self):
        return len(self.buffer)


class NStepBuffer:
    def __init__(self, n=N_STEP, gamma=GAMMA):
        self.n, self.gamma = n, gamma
        self.buffer = deque(maxlen=n)

    def push(self, transition):
        self.buffer.append(transition)

    def get(self):
        if len(self.buffer) < self.n:
            return None
        s0, a0 = self.buffer[0][0], self.buffer[0][1]
        r = sum(self.gamma ** i * self.buffer[i][2] for i in range(self.n))
        return (s0, a0, r, self.buffer[-1][3],
                self.buffer[-1][4])

    def flush(self):
        results = []
        while self.buffer:
            s0, a0 = self.buffer[0][0], self.buffer[0][1]
            r = sum(self.gamma ** i * self.buffer[i][2]
                    for i in range(len(self.buffer)))
            results.append(
                (s0, a0, r, self.buffer[-1][3],
                 self.buffer[-1][4]))
            self.buffer.popleft()
        return results

    def reset(self):
        self.buffer.clear()


class StrategyPool:
    def __init__(self, max_size=8):
        self.max_size = max_size
        self.pool = []
        self.generation = 0

    def add(self, name, state_dict, fitness):
        self.pool.append(
            (name, copy.deepcopy(state_dict), fitness))
        self.generation += 1
        if len(self.pool) > self.max_size:
            self.pool.sort(key=lambda x: x[2], reverse=True)
            self.pool = self.pool[:self.max_size]

    def sample_opponent(self):
        if not self.pool:
            return None
        if random.random() < 0.7:
            return self.best()
        return random.choice(self.pool)

    def best(self):
        if not self.pool:
            return None
        return max(self.pool, key=lambda x: x[2])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {"pool": self.pool, "gen": self.generation}, f)

    def load(self, path):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self.pool = data.get("pool", [])
                self.generation = data.get("gen", 0)
                print(f"  🧬 Pool: {len(self.pool)} strategies, "
                      f"gen {self.generation}")
            except Exception as e:
                print(f"  ⚠ Pool load error: {e}")


# ╔════════════════════════════════════════════╗
# ║             规则AI                         ║
# ╚════════════════════════════════════════════╝

def _rule_ai_logic(world, me, enemy):
    in_danger = False
    for b in world.bombs:
        if ((b.x == me.x and abs(b.y - me.y) <= b.power)
                or (b.y == me.y
                    and abs(b.x - me.x) <= b.power)):
            if b.timer <= 4:
                in_danger = True
                break
    if in_danger:
        best_dir, best_safety = 5, -1
        for act in range(4):
            dx, dy = DIR_MAP[act]
            nx, ny = me.x + dx, me.y + dy
            if not world._can_move(nx, ny):
                continue
            if nx == enemy.x and ny == enemy.y:
                continue
            safe, safety = True, 0
            for b in world.bombs:
                if ((b.x == nx and abs(b.y - ny) <= b.power)
                        or (b.y == ny
                            and abs(b.x - nx) <= b.power)):
                    safe = False
                safety += abs(b.x - nx) + abs(b.y - ny)
            safety += 10 if safe else 0
            if safety > best_safety:
                best_safety, best_dir = safety, act
        return best_dir
    dist = abs(me.x - enemy.x) + abs(me.y - enemy.y)
    if (dist <= 3 and me.bomb_cooldown <= 0
            and me.active_bombs < me.max_bombs):
        for act in range(4):
            dx, dy = DIR_MAP[act]
            nx, ny = me.x + dx, me.y + dy
            if (world._can_move(nx, ny)
                    and not (nx == enemy.x and ny == enemy.y)):
                return 4
    best_dir, best_dist = 5, dist
    for act in range(4):
        dx, dy = DIR_MAP[act]
        nx, ny = me.x + dx, me.y + dy
        if not world._can_move(nx, ny):
            continue
        if nx == enemy.x and ny == enemy.y:
            continue
        nd = abs(nx - enemy.x) + abs(ny - enemy.y)
        if nd < best_dist:
            best_dist, best_dir = nd, act
    return best_dir


def rule_based_ai(world):
    return _rule_ai_logic(world, world.ai, world.player)


def rule_based_player(world):
    act = _rule_ai_logic(world, world.player, world.ai)
    if random.random() < 0.12:
        act = random.randint(0, 5)
    return act


# ╔════════════════════════════════════════════╗
# ║          AMP Context                       ║
# ╚════════════════════════════════════════════╝

class NullContext:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass


class AMPContext:
    def __init__(self, use_amp, device):
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = (torch.amp.GradScaler("cuda")
                       if self.use_amp else None)

    def autocast(self):
        if self.use_amp:
            return torch.amp.autocast("cuda")
        return NullContext()

    def scale_and_step(self, loss, optimizer, params,
                       max_norm=10.0):
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, max_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm)
            optimizer.step()


# ╔════════════════════════════════════════════╗
# ║          UI 小组件 (GUI模式)               ║
# ╚════════════════════════════════════════════╝

class MiniChart:
    def __init__(self, x, y, w, h, title, color, max_pts=150):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.title = title
        self.color = color
        self.data = deque(maxlen=max_pts)

    def add(self, v):
        self.data.append(v)

    def draw(self, surf, font):
        import pygame
        pygame.draw.rect(surf, (10, 10, 22), (self.x, self.y, self.w, self.h))
        pygame.draw.rect(surf, (40, 40, 60),
                         (self.x, self.y, self.w, self.h), 1)
        surf.blit(font.render(self.title, True, C_DIM), (self.x + 4, self.y + 2))
        if len(self.data) < 2:
            return
        dl = list(self.data)
        mn, mx = min(dl), max(dl)
        rng = mx - mn if mx != mn else 1.0
        cy = self.y + 15
        ch = self.h - 18
        surf.blit(font.render(f"{dl[-1]:.2f}", True, self.color),
                  (self.x + self.w - 52, self.y + 2))
        win = min(30, len(dl))
        avg = []
        for i in range(len(dl)):
            s = max(0, i - win + 1)
            avg.append(sum(dl[s:i + 1]) / (i - s + 1))
        pts, apts = [], []
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
            dim_c = tuple(v // 3 for v in self.color)
            pygame.draw.lines(surf, dim_c, False, pts, 1)
        if len(apts) >= 2:
            pygame.draw.lines(surf, self.color, False, apts, 2)


class QValueBar:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.q_values = np.zeros(ACTION_DIM)

    def update(self, q_values):
        self.q_values = q_values

    def draw(self, surf, font):
        import pygame
        pygame.draw.rect(surf, (10, 10, 22),
                         (self.x, self.y, self.w, self.h))
        pygame.draw.rect(surf, (40, 40, 60),
                         (self.x, self.y, self.w, self.h), 1)
        surf.blit(font.render("Q-Values", True, C_DIM),
                  (self.x + 4, self.y + 2))
        if np.all(self.q_values == 0):
            return
        bar_area_y = self.y + 15
        bar_h = self.h - 18
        bar_w = (self.w - 12) / ACTION_DIM - 2
        q_min, q_max = self.q_values.min(), self.q_values.max()
        q_range = q_max - q_min if q_max != q_min else 1.0
        best_a = self.q_values.argmax()
        for i in range(ACTION_DIM):
            bx = self.x + 6 + i * (bar_w + 2)
            norm = (self.q_values[i] - q_min) / q_range
            bh = max(2, int(norm * (bar_h - 12)))
            by = bar_area_y + bar_h - bh - 2
            color = C_GOOD if i == best_a else (60, 60, 90)
            pygame.draw.rect(surf, color,
                             (int(bx), int(by), int(bar_w), bh))
            label = DIR_NAMES[i][0]
            txt = font.render(label, True, C_DIM)
            surf.blit(txt, (int(bx + bar_w // 2 - txt.get_width() // 2),
                            int(bar_area_y + bar_h - 12)))


class Particle:
    __slots__ = ["x", "y", "vx", "vy", "life", "max_life",
                 "color", "size"]
    def __init__(self, x, y, color, speed_range=(1, 4)):
        self.x, self.y = float(x), float(y)
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
        import pygame
        a = max(0.0, self.life / self.max_life)
        r = int(self.size * a)
        if r > 0:
            c = tuple(min(255, int(v * a)) for v in self.color)
            pygame.draw.circle(
                surf, c, (int(self.x), int(self.y)), r)


# ╔════════════════════════════════════════════╗
# ║              主渲染器                      ║
# ╚════════════════════════════════════════════╝

class Renderer:
    def __init__(self, screen, clock, fonts, hw_tier):
        self.screen = screen
        self.clock = clock
        self.fonts = fonts
        self.hw_tier = hw_tier
        self.particles = []
        self.pulse = 0.0
        px = ARENA_W + 10
        cw = PANEL_W - 20
        self.chart_winrate = MiniChart(
            px, 10, cw, 58, "AI WinRate%", C_CHART_WIN)
        self.chart_reward = MiniChart(
            px, 74, cw, 58, "Reward", C_CHART_REW)
        self.chart_eps = MiniChart(
            px, 138, cw, 58, "Epsilon", C_CHART_EPS)
        self.chart_loss = MiniChart(
            px, 202, cw, 58, "Loss", C_CHART_LOSS)
        self.qbar = QValueBar(px, 266, cw, 58)

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
        import pygame
        self.pulse = (self.pulse + 0.1) % (2 * math.pi)
        dt = 1.0 / max(BASE_FPS, 15)
        self.screen.fill(C_BG)
        dmap = world.get_danger_map()
        for y in range(ARENA_ROWS):
            for x in range(ARENA_COLS):
                rx, ry = x * CELL, y * CELL
                cell = world.grid[y][x]
                if cell == WALL:
                    pygame.draw.rect(self.screen, C_WALL,
                                     (rx, ry, CELL, CELL))
                    pygame.draw.rect(self.screen, C_WALL_L,
                                     (rx+1, ry+1, CELL-2, CELL-2), 1)
                elif cell == BRICK:
                    pygame.draw.rect(self.screen, C_BRICK,
                                     (rx, ry, CELL, CELL))
                    pygame.draw.rect(self.screen, C_BRICK_D,
                                     (rx+2, ry+2, CELL-4, CELL-4))
                    pygame.draw.line(self.screen, C_BRICK_D,
                                     (rx, ry + CELL//2),
                                     (rx + CELL, ry + CELL//2), 1)
                    pygame.draw.line(self.screen, C_BRICK_D,
                                     (rx + CELL//2, ry),
                                     (rx + CELL//2, ry + CELL), 1)
                else:
                    pygame.draw.rect(self.screen, C_FLOOR,
                                     (rx, ry, CELL, CELL))
                    d = dmap[y][x]
                    if d > 0.01:
                        ds = pygame.Surface(
                            (CELL, CELL), pygame.SRCALPHA)
                        ds.fill((255, 50, 0, int(d * 100)))
                        self.screen.blit(ds, (rx, ry))
                pygame.draw.rect(self.screen, C_GRID,
                                 (rx, ry, CELL, CELL), 1)
        #道具
        pu_icons = ["R", "S", "B"]
        pu_colors = [C_POWERUP, C_WARN, C_SHIELD]
        for pu in world.powerups:
            px_ = pu.x * CELL + CELL // 2
            py_ = pu.y * CELL + CELL // 2
            r = int(CELL * 0.3 + math.sin(self.pulse * 2) * 2)
            c = pu_colors[pu.kind % 3]
            pygame.draw.circle(
                self.screen, c, (px_, py_), max(r, 4))
            txt = self.fonts["sm"].render(
                pu_icons[pu.kind % 3], True, (0, 0, 0))
            self.screen.blit(
                txt, (px_ - txt.get_width() // 2, py_ - txt.get_height() // 2))
        #炸弹
        for b in world.bombs:
            bx = b.x * CELL + CELL // 2
            by = b.y * CELL + CELL // 2
            urgency = 1.0 - b.timer / BOMB_TIMER
            r = int(CELL * 0.35 + urgency * 4)
            flash = b.timer <= 3 and b.timer % 2 == 0
            bc = C_BOMB_FUSE if flash else C_BOMB
            pygame.draw.circle(
                self.screen, bc, (bx, by), max(r, 4))
            pygame.draw.line(self.screen, C_BOMB_FUSE,
                             (bx, by - r),
                             (bx + 3, by - r - 5), 2)
            txt = self.fonts["sm"].render(
                str(b.timer), True, (0, 0, 0))
            self.screen.blit(
                txt, (bx - txt.get_width() // 2,
                      by - txt.get_height() // 2))
        # 爆炸
        for exp in world.explosions:
            ex, ey = exp.x * CELL, exp.y * CELL
            ci = min(exp.timer, len(C_EXPLODE) - 1)
            pygame.draw.rect(self.screen, C_EXPLODE[ci],
                             (ex+2, ey+2, CELL-4, CELL-4))
        # 角色
        self._draw_fighter(world.player, C_PLAYER,
                           C_PLAYER_D, "P")
        self._draw_fighter(world.ai, C_AI, C_AI_D, "AI")
        # 粒子
        i = 0
        while i < len(self.particles):
            if self.particles[i].update(dt):
                self.particles[i].draw(self.screen)
                i += 1
            else:
                self.particles.pop(i)
        pygame.draw.rect(self.screen, (80, 80, 120),
                         (0, 0, ARENA_W, ARENA_H), 3)

    def _draw_fighter(self, fighter, color, dark_color, label):
        import pygame
        if fighter.hp <= 0:
            return
        fx = fighter.x * CELL + CELL // 2
        fy = fighter.y * CELL + CELL // 2
        r = CELL // 2 - 3
        c = color
        if fighter.invincible > 0 and fighter.invincible % 2 == 0:
            c = (255, 255, 255)
        pygame.draw.circle(self.screen, dark_color, (fx, fy), r)
        pygame.draw.circle(self.screen, c, (fx, fy), r - 2)
        txt = self.fonts["sm"].render(label, True, (0, 0, 0))
        self.screen.blit(
            txt, (fx - txt.get_width() // 2,
                  fy - txt.get_height() // 2))
        hp_w = CELL - 6
        hp_h = 4
        hp_x = fighter.x * CELL + 3
        hp_y = fighter.y * CELL - 6
        pygame.draw.rect(self.screen, (40, 40, 40),
                         (hp_x, hp_y, hp_w, hp_h))
        fill = int(hp_w * fighter.hp / fighter.max_hp)
        hc = C_GOOD if fighter.hp > 1 else C_BAD
        pygame.draw.rect(self.screen, hc,
                         (hp_x, hp_y, fill, hp_h))
        act = fighter.last_action
        if act < 4:
            dx, dy = DIR_MAP[act]
            ax1, ay1 = fx + dx * 8, fy + dy * 8
            ax2, ay2 = fx + dx * 16, fy + dy * 16
            pygame.draw.line(self.screen, c,
                             (ax1, ay1), (ax2, ay2), 2)
            pygame.draw.circle(self.screen, c, (ax2, ay2), 3)
        elif act == 4:
            pygame.draw.circle(
                self.screen, C_BOMB, (fx, fy - r - 6), 4)

    def draw_panel(self, world, episode, epsilon, loss, mode,
                   speed, ai_wins, player_wins, total_rounds,
                   gate_val, strategy_gen, fps_val, lr_val,
                   streak, best_streak, is_eval):
        import pygame
        px = ARENA_W
        pygame.draw.rect(self.screen, C_PANEL,
                         (px, 0, PANEL_W, ARENA_H))
        pygame.draw.line(self.screen, (50, 50, 80),
                         (px, 0), (px, ARENA_H), 2)
        self.chart_winrate.draw(self.screen, self.fonts["sm"])
        self.chart_reward.draw(self.screen, self.fonts["sm"])
        self.chart_eps.draw(self.screen, self.fonts["sm"])
        self.chart_loss.draw(self.screen, self.fonts["sm"])
        self.qbar.draw(self.screen, self.fonts["sm"])
        iy = 332
        ipx = px + 10
        hearts_p = ("♥" * world.player.hp
                    + "·" * (MAX_HP - world.player.hp))
        hearts_a = ("♥" * world.ai.hp
                    + "·" * (MAX_HP - world.ai.hp))
        wr = ai_wins / max(total_rounds, 1) * 100
        mode_str = mode
        if is_eval:
            mode_str += " [EVAL]"
        infos = [
            ("Mode", mode_str, C_HIGHLIGHT if mode == "PvAI" else C_GOOD),
            ("Round", f"{episode}", C_TEXT),
            ("Step",
             f"{world.step_count}/{MAX_ROUND_STEPS}", C_TEXT),
            ("P HP", hearts_p, C_PLAYER),
            ("AI HP", hearts_a, C_AI),
            ("", "", C_TEXT),
            ("P Wins", f"{player_wins}", C_PLAYER),
            ("AI Wins", f"{ai_wins}", C_AI),
            ("WinRate", f"{wr:.1f}%", C_WARN),
            ("Streak",
             f"{streak} (best:{best_streak})", C_GOOD),
            ("", "", C_TEXT),
            ("Epsilon", f"{epsilon:.4f}", C_TEXT),
            ("Loss", f"{loss:.5f}", C_TEXT),
            ("LR", f"{lr_val:.6f}", C_DIM),
            ("Gate", f"{gate_val:.2f}", C_WARN),
            ("Gen", f"{strategy_gen}", C_GOOD),
            ("Speed", f"x{speed}", C_TEXT),
            ("FPS", f"{fps_val:.0f}", C_DIM),
        ]
        for lbl, val, color in infos:
            if lbl:
                self.screen.blit(
                    self.fonts["sm"].render(
                        f"{lbl}:", True, C_DIM),
                    (ipx, iy))
                self.screen.blit(
                    self.fonts["sm"].render(
                        str(val), True, color),
                    (ipx + 60, iy))
            iy += 13

    def draw_bottom(self, mode, epsilon, global_step, device_str):
        import pygame
        by = ARENA_H
        pygame.draw.rect(self.screen, C_BOTTOM,
                         (0, by, WIN_W, BOTTOM_H))
        pygame.draw.line(self.screen, (50, 50, 80),
                         (0, by), (WIN_W, by), 2)
        y1, y2, y3, y4, y5 = (
            by + 6, by + 22, by + 38, by + 54, by + 68)
        self.screen.blit(
            self.fonts["lg"].render(
                f"Grid Duel Arena v{VERSION}[{self.hw_tier}]",
                True, C_HIGHLIGHT), (10, y1))
        self.screen.blit(
            self.fonts["sm"].render(
                "[Space]Pause [Up/Dn]Speed "
                "[1]PvAI [2]SelfPlay [3]Train",
                True, C_DIM), (10, y2))
        self.screen.blit(
            self.fonts["sm"].render(
                "[WASD]Move [J]Bomb [K]Stay "
                "[Ctrl+S]Save [Tab]Help [Esc]Quit",
                True, C_DIM), (10, y3))
        self.screen.blit(
            self.fonts["sm"].render(
                f"Mode:{mode} | {device_str}",
                True, C_DIM), (10, y4))
        prog_x, prog_w, prog_h, prog_y = 10, WIN_W - 20, 6, y5
        pygame.draw.rect(self.screen, (30, 30, 50),
                         (prog_x, prog_y, prog_w, prog_h))
        progress = min(1.0, global_step / EPS_DECAY)
        fill_w = int(prog_w * progress)
        bar_color = (C_GOOD if progress > 0.8
                     else C_WARN if progress > 0.4
                     else C_BAD)
        pygame.draw.rect(self.screen, bar_color,
                         (prog_x, prog_y, fill_w, prog_h))
        pct_txt = self.fonts["sm"].render(
            f"Train: {progress*100:.0f}% (ε={epsilon:.3f})",
            True, C_DIM)
        self.screen.blit(
            pct_txt,
            (prog_x + prog_w + 5 - pct_txt.get_width(),
             prog_y - 10))

    def draw_round_result(self, winner, p_wins, ai_wins, total):
        import pygame
        ov = pygame.Surface(
            (ARENA_W, ARENA_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 160))
        self.screen.blit(ov, (0, 0))
        cx, cy = ARENA_W // 2, ARENA_H // 2
        if winner == "player":
            txt, color = "PLAYER WINS!", C_PLAYER
        elif winner == "ai":
            txt, color = "AI WINS!", C_AI
        else:
            txt, color = "DRAW!", C_HIGHLIGHT
        title = self.fonts["xl"].render(txt, True, color)
        self.screen.blit(
            title, (cx - title.get_width() // 2, cy - 50))
        score = self.fonts["md"].render(
            f"Player {p_wins} : {ai_wins} AI(of {total})",
            True, C_TEXT)
        self.screen.blit(
            score, (cx - score.get_width() // 2, cy + 10))
        hint = self.fonts["sm"].render(
            "[N] Next Round  |  [Esc] Quit", True, C_DIM)
        self.screen.blit(
            hint, (cx - hint.get_width() // 2, cy + 40))

    def draw_help_overlay(self, device_str, hidden_dim,
                          batch_size, memory_size, use_amp,
                          grad_accum, warmup_steps):
        import pygame
        ov = pygame.Surface(
            (WIN_W, WIN_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 200))
        self.screen.blit(ov, (0, 0))
        lines = [
            (f"Grid Duel Arena v{VERSION}",
             C_HIGHLIGHT, "lg"),
            ("", C_TEXT, "sm"),
            ("--- Controls ---", C_WARN, "md"),
            ("[W/A/S/D] Move player", C_TEXT, "md"),
            ("[J] Place bomb  [K] Stay", C_TEXT, "md"),
            ("[Space] Pause  [Up/Down] Speed",
             C_TEXT, "md"),
            ("[1] PvAI  [2] SelfPlay  [3] FastTrain",
             C_TEXT, "md"),
            ("[Ctrl+S] Save checkpoint", C_TEXT, "md"),
            ("[Tab] Toggle this help", C_TEXT, "md"),
            ("[Esc] Save & quit", C_TEXT, "md"),
            ("", C_TEXT, "sm"),
            ("--- Architecture ---", C_WARN, "md"),
            ("Dual-channel ReflexNet (Fast+Slow)",
             C_TEXT, "md"),
            (f"  Fast: 18->48->24->{ACTION_DIM}",
             C_DIM, "sm"),
            (f"  Slow: {STATE_DIM}->{hidden_dim}->..."
             f"->{ACTION_DIM}", C_DIM, "sm"),
            ("  Dueling DQN + N-step + PER",
             C_DIM, "sm"),
            ("", C_TEXT, "sm"),
            ("--- Hardware ---", C_WARN, "md"),
            (f"Device: {device_str}Tier: {self.hw_tier}",
             C_TEXT, "md"),
            (f"Hidden: {hidden_dim}  Batch: {batch_size}  "
             f"Memory: {memory_size}", C_DIM, "sm"),
            (f"AMP: {use_amp}  GradAccum: {grad_accum}  "
             f"Warmup: {warmup_steps}", C_DIM, "sm"),
            ("", C_TEXT, "sm"),
            ("Press [Tab] to close", C_HIGHLIGHT, "md"),
        ]
        y = 20
        for text, color, font_key in lines:
            if text:
                self.screen.blit(
                    self.fonts[font_key].render(
                        text, True, color), (30, y))
            y += 18 if font_key != "lg" else 26


# ╔════════════════════════════════════════════╗
# ║          存档/加载                         ║
# ╚════════════════════════════════════════════╝

def migrate_weights(model, old_sd, label=""):
    ns = model.state_dict()
    matched, used = 0, set()
    for k in ns:
        if k in old_sd and old_sd[k].shape == ns[k].shape:
            ns[k] = old_sd[k]
            matched += 1
            used.add(k)
    unmatched_new = [k for k in ns
                     if k not in used and k not in old_sd]
    unmatched_old = [k for k in old_sd if k not in used]
    if unmatched_new and unmatched_old:
        def pfx(k):
            p = k.rsplit(".", 2)
            return p[0] if len(p) >= 3 else ""
        def sfx(k):
            return k.rsplit(".", 1)[-1]
        npfx = defaultdict(list)
        opfx = defaultdict(list)
        for k in unmatched_new:
            npfx[pfx(k)].append(k)
        for k in unmatched_old:
            opfx[pfx(k)].append(k)
        for p in npfx:
            if p not in opfx:
                continue
            for nk in sorted(npfx[p]):
                for ok in sorted(opfx[p]):
                    if ok in used:
                        continue
                    if (sfx(ok) == sfx(nk)
                            and old_sd[ok].shape == ns[nk].shape):
                        ns[nk] = old_sd[ok]
                        matched += 1
                        used.add(ok)
                        break
    skipped = len(ns) - matched
    model.load_state_dict(ns)
    if skipped > 0:
        print(f"  🔄 {label}: {matched} matched, "
              f"{skipped} re-initialized")
    else:
        print(f"  ✅ {label}: all {matched} layers matched")
    return matched, skipped


def save_checkpoint(net, target_net, optimizer, scheduler,
                    memory, stats, pool, episode, best_reward,
                    hidden_dim, device):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({
        "version": VERSION,
        "episode": episode,
        "best_reward": best_reward,
        "hidden_dim": hidden_dim,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "net": net.state_dict(),
        "target": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, CKPT_MODEL)
    stats_save = {}
    for k, v in stats.items():
        stats_save[k] = v[-5000:] if isinstance(v, list) else v
    with open(CKPT_STATS, "w") as f:
        json.dump(stats_save, f, indent=2)
    pool.save(CKPT_POOL)
    print(f"  💾 Saved ep={episode} best_r={best_reward:.1f}")


def save_best(net, best_reward, episode, hidden_dim):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({
        "version": VERSION, "episode": episode,
        "best_reward": best_reward,
        "hidden_dim": hidden_dim,
        "net": net.state_dict(),
    }, CKPT_BEST)
    print(f"🏆 Best model R={best_reward:.1f}")


def load_checkpoint(net, target_net, optimizer, scheduler,
                    pool, hidden_dim, device):
    default_stats = {
        "ai_wins": 0, "player_wins": 0, "draws": 0,
        "rewards": [], "winrates": [], "losses": [],
        "best_streak": 0,}
    if not os.path.exists(CKPT_MODEL):
        print("  🆕 No checkpoint — starting fresh")
        return 0, -1e9, default_stats
    print(f"  📂 Loading {CKPT_MODEL}...")
    ckpt = torch.load(
        CKPT_MODEL, map_location=device, weights_only=False)
    old_ver = ckpt.get("version", "1.0")
    old_hidden = ckpt.get("hidden_dim", 128)
    print(f"  📋 Checkpoint: v{old_ver}, hidden={old_hidden}, "
          f"current: v{VERSION}, hidden={hidden_dim}")
    if old_hidden == hidden_dim and old_ver == VERSION:
        net.load_state_dict(ckpt["net"])
        target_net.load_state_dict(
            ckpt.get("target", ckpt["net"]))
        print(f"  ✅ Weights loaded perfectly")
    else:
        reason = (f"hidden {old_hidden}→{hidden_dim}"
                  if old_hidden != hidden_dim
                  else f"v{old_ver}→v{VERSION}")
        print(f"  🔄 Architecture changed ({reason}), "
              f"migrating...")
        migrate_weights(net, ckpt["net"], "net")
        migrate_weights(
            target_net,
            ckpt.get("target", ckpt["net"]), "target")
    if old_hidden == hidden_dim and old_ver == VERSION:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            print("  ⚠ Optimizer re-initialized")
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                print("  ⚠ Scheduler re-initialized")
    pool.load(CKPT_POOL)
    stats = dict(default_stats)
    if os.path.exists(CKPT_STATS):
        try:
            with open(CKPT_STATS) as f:
                loaded = json.load(f)
            for k in stats:
                if k in loaded:
                    stats[k] = loaded[k]
        except Exception as e:
            print(f"  ⚠ Stats load error: {e}")
    ep = ckpt.get("episode", 0)
    best = ckpt.get("best_reward", -1e9)
    print(f"  ✅ Resumed: ep={ep}, best={best:.1f}")
    return ep, best, stats


# ╔════════════════════════════════════════════╗
# ║      硬件检测(只在主进程实例化)            ║
# ╚════════════════════════════════════════════╝

class HardwareProfile:
    def __init__(self, forced_device="auto"):
        self.has_cuda = torch.cuda.is_available()
        self.gpu_name = ""
        self.vram_mb = 0
        self.cpu_cores = os.cpu_count() or 2
        self.ram_mb = self._get_ram()
        self.tier = "cpu_low"
        self.device = torch.device("cpu")
        if forced_device == "cpu":
            self.has_cuda = False
        elif forced_device == "cuda" and not self.has_cuda:
            print("  ⚠ CUDA requested but not available")
        if self.has_cuda:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.vram_mb = (
                torch.cuda.get_device_properties(0)
                .total_memory // (1024 * 1024))
            self._classify_gpu()
        else:
            self._classify_cpu()
        self.config = self._build_config()
        self._print_report()

    def _get_ram(self):
        try:
            if sys.platform == "linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if "MemTotal" in line:
                            return int(line.split()[1]) // 1024
            elif sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ull = ctypes.c_ulonglong
                class MEMSTAT(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", c_ull),
                        ("ullAvailPhys", c_ull),
                        ("ullTotalPageFile", c_ull),
                        ("ullAvailPageFile", c_ull),
                        ("ullTotalVirtual", c_ull),
                        ("ullAvailVirtual", c_ull),
                        ("sullAvailExtVirt", c_ull)]
                stat = MEMSTAT()
                stat.dwLength = ctypes.sizeof(stat)
                kernel32.GlobalMemoryStatusEx(
                    ctypes.byref(stat))
                return stat.ullTotalPhys // (1024 * 1024)
            elif sys.platform == "darwin":
                import subprocess
                out = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"])
                return int(out.strip()) // (1024 * 1024)
        except Exception:
            pass
        return 4096

    def _classify_gpu(self):
        if self.vram_mb >= 8000:
            self.tier = "gpu_large"
            self.device = torch.device("cuda")
        elif self.vram_mb >= 4000:
            self.tier = "gpu_medium"
            self.device = torch.device("cuda")
        elif self.vram_mb >= 2000:
            self.tier = "gpu_small"
            self.device = torch.device("cuda")
        elif self.vram_mb >= 1000:
            self.tier = "gpu_tiny"
            self.device = torch.device("cuda")
        else:
            self.tier = "cpu_fallback"

    def _classify_cpu(self):
        if self.ram_mb >= 8000 and self.cpu_cores >= 4:
            self.tier = "cpu_high"
        elif self.ram_mb >= 4000:
            self.tier = "cpu_mid"
        else:
            self.tier = "cpu_low"

    def _build_config(self):
        profiles = {
            "gpu_large": dict(
                hidden_dim=192, batch_size=256,
                memory_size=50000,
                train_steps_per_frame=2,
                max_fps_train=300,
                use_amp=True, grad_accum=1,
                warmup_steps=500),
            "gpu_medium": dict(
                hidden_dim=160, batch_size=128,
                memory_size=30000,
                train_steps_per_frame=2,
                max_fps_train=240,
                use_amp=True, grad_accum=1,
                warmup_steps=400),
            "gpu_small": dict(
                hidden_dim=128, batch_size=96,
                memory_size=20000,
                train_steps_per_frame=1,
                max_fps_train=180,
                use_amp=True, grad_accum=1,
                warmup_steps=300),
            "gpu_tiny": dict(
                hidden_dim=96, batch_size=48,
                memory_size=12000,
                train_steps_per_frame=1,
                max_fps_train=120,
                use_amp=False, grad_accum=2,
                warmup_steps=200),
            "cpu_high": dict(
                hidden_dim=128, batch_size=128,
                memory_size=30000,
                train_steps_per_frame=4,
                max_fps_train=120,
                use_amp=False, grad_accum=1,
                warmup_steps=300),
            "cpu_mid": dict(
                hidden_dim=96, batch_size=48,
                memory_size=10000,
                train_steps_per_frame=1,
                max_fps_train=90,
                use_amp=False, grad_accum=2,
                warmup_steps=200),
            "cpu_low": dict(
                hidden_dim=64, batch_size=32,
                memory_size=8000,
                train_steps_per_frame=1,
                max_fps_train=60,
                use_amp=False, grad_accum=2,
                warmup_steps=150),
            "cpu_fallback": dict(
                hidden_dim=64, batch_size=32,
                memory_size=8000,
                train_steps_per_frame=1,
                max_fps_train=60,
                use_amp=False, grad_accum=2,
                warmup_steps=150),
        }
        return profiles.get(self.tier, profiles["cpu_low"])

    def _print_report(self):
        print("\n" + "=" * 56)
        print("  ⚙️  Hardware Auto-Detection Report")
        print("=" * 56)
        if self.has_cuda:
            print(f"  GPU : {self.gpu_name}")
            print(f"  VRAM: {self.vram_mb} MB")
        else:
            print("  GPU : Not available (CPU only)")
        print(f"  CPU : {self.cpu_cores} cores")
        print(f"  RAM : {self.ram_mb} MB")
        print(f"  Tier: {self.tier}")
        print(f"  Device: {self.device}")
        print("-" * 56)
        for k, v in self.config.items():
            print(f"  {k:22s}: {v}")
        print("=" * 56 + "\n")


# ╔════════════════════════════════════════════╗
# ║      ★★★ Worker进程函数 (修复核心) ★★★     ║
# ╚════════════════════════════════════════════╝

def _env_worker(worker_id, transition_queue, result_queue,
                weight_dict, weight_version, stop_flag,
                config):
    """
    环境工作进程 — 不触发任何模块级初始化
    整局数据攒好后一次性 put 到 Queue
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    local_device = torch.device("cpu")
    local_net = ReflexNet(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        hidden=config["hidden_dim"]
    ).to(local_device)
    local_net.eval()

    local_weight_ver = -1
    local_episode = 0
    rng = random.Random(
        worker_id * 10000 + int(time.time()))
    nstep = NStepBuffer(N_STEP, GAMMA)

    def sync_weights():
        nonlocal local_weight_ver
        try:
            ver = weight_version.value
            if ver > local_weight_ver and "net" in weight_dict:
                local_net.load_state_dict(weight_dict["net"])
                local_weight_ver = ver
        except Exception:
            pass

    while not stop_flag.value:
        sync_weights()
        local_episode += 1
        global_step_est = local_episode * 250
        epsilon = max(
            config["eps_end"],
            config["eps_start"]
            - global_step_est / config["eps_decay"])

        seed = rng.randint(0, 2**31)
        world = DuelArena(seed=seed)
        obs = world.reset()
        nstep.reset()
        total_reward = 0.0
        episode_transitions = []

        while not world.round_over and not stop_flag.value:
            p_act = rule_based_player(world)
            with torch.no_grad():
                obs_t = torch.tensor(
                    obs, dtype=torch.float32,
                    device=local_device).unsqueeze(0)
                q_values = local_net(obs_t)
            if rng.random() < epsilon:
                ai_act = rng.randint(0, ACTION_DIM - 1)
            else:
                ai_act = q_values.argmax(dim=-1).item()

            obs2, reward, done = world.step(p_act, ai_act)
            total_reward += reward
            nstep.push(
                (obs, ai_act, reward, obs2, float(done)))
            nt = nstep.get()
            if nt is not None:
                episode_transitions.append(nt)
            obs = obs2
            if done:
                for t in nstep.flush():
                    episode_transitions.append(t)
                break

        if stop_flag.value:
            break

        #★ 一次性发送整局
        if episode_transitions:
            states = np.array(
                [t[0] for t in episode_transitions],
                dtype=np.float32)
            actions = np.array(
                [t[1] for t in episode_transitions],
                dtype=np.int64)
            rewards = np.array(
                [t[2] for t in episode_transitions],
                dtype=np.float32)
            next_states = np.array(
                [t[3] for t in episode_transitions],
                dtype=np.float32)
            dones = np.array(
                [t[4] for t in episode_transitions],
                dtype=np.float32)
            try:
                transition_queue.put(
                    (states, actions, rewards,
                     next_states, dones),
                    timeout=1.0)
            except Exception:
                pass

        try:
            result_queue.put_nowait(
                (world.winner, total_reward,
                 world.step_count))
        except Exception:
            pass


# ╔════════════════════════════════════════════╗
# ║      ★★★ 并行收集器 (修复核心) ★★★         ║
# ╚════════════════════════════════════════════╝

class ParallelCollector:
    def __init__(self, num_workers, config, net):
        self.num_workers = num_workers
        self.config = config
        self.workers = []
        self.running = False
        self.manager = mp.Manager()
        self.transition_queue = mp.Queue(maxsize=500)
        self.result_queue = mp.Queue(maxsize=2000)
        self.weight_dict = self.manager.dict()
        self.weight_version = mp.Value("i", 0)
        self.stop_flag = mp.Value("b", False)
        self.broadcast_weights(net)

    def broadcast_weights(self, net):
        cpu_state = {
            k: v.cpu().clone()
            for k, v in net.state_dict().items()}
        self.weight_dict["net"] = cpu_state
        self.weight_version.value += 1

    def start(self):
        worker_config = {
            "hidden_dim": self.config["hidden_dim"],
            "eps_start": EPS_START,
            "eps_end": EPS_END,
            "eps_decay": EPS_DECAY,
        }
        self.stop_flag.value = False
        for i in range(self.num_workers):
            p = mp.Process(
                target=_env_worker,
                args=(i, self.transition_queue,
                      self.result_queue,
                      self.weight_dict,
                      self.weight_version,
                      self.stop_flag,
                      worker_config),
                daemon=True)
            p.start()
            self.workers.append(p)
        self.running = True
        print(f"  🚀 Started {self.num_workers} env workers "
              f"(PIDs: {[p.pid for p in self.workers[:5]]}...)")

    def collect_transitions(self, max_batches=200):
        all_transitions = []
        results = []
        batches = 0
        while batches < max_batches:
            try:
                batch_data = (
                    self.transition_queue.get_nowait())
                states, actions, rewards, ns, dones = batch_data
                n = len(states)
                for i in range(n):
                    all_transitions.append((
                        states[i], actions[i], rewards[i],
                        ns[i], dones[i]))
                batches += 1
            except Exception:
                break
        while True:
            try:
                r = self.result_queue.get_nowait()
                results.append(r)
            except Exception:
                break
        return all_transitions, results

    def stop(self):
        self.stop_flag.value = True
        time.sleep(0.5)
        for p in self.workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        self.workers.clear()
        self.running = False
        for q in [self.transition_queue, self.result_queue]:
            while True:
                try:
                    q.get_nowait()
                except Exception:
                    break
        print("⏹️  All workers stopped")

    def get_queue_size(self):
        try:
            return self.transition_queue.qsize()
        except Exception:
            return 0


# ╔════════════════════════════════════════════╗
# ║          命令行参数                        ║
# ╚════════════════════════════════════════════╝

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid Duel Arena v3.2")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def detect_display():
    if sys.platform == "linux":
        if (not os.environ.get("DISPLAY")
                and not os.environ.get("WAYLAND_DISPLAY")):
            return False
    elif sys.platform in ("win32", "darwin"):
        return True
    try:
        os.environ.setdefault(
            "PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        pygame.init()
        info = pygame.display.Info()
        if info.current_w <= 0 or info.current_h <= 0:
            pygame.quit()
            return False
        pygame.quit()
        return True
    except Exception:
        return False


# ╔════════════════════════════════════════════╗
# ║      无头训练主循环 (v3.2修复版)           ║
# ╚════════════════════════════════════════════╝

def main_headless():
    args = parse_args()
    hw = HardwareProfile(forced_device=args.device)
    DEVICE = hw.device
    HW_CFG = hw.config
    # 👇 新增下面这一行，强行锁死隐藏层维度，放弃自适应
    HW_CFG["hidden_dim"] = 128
    HIDDEN_DIM = HW_CFG["hidden_dim"]
    BATCH_SIZE = HW_CFG["batch_size"]
    MEMORY_SIZE = HW_CFG["memory_size"]
    USE_AMP = HW_CFG["use_amp"]
    WARMUP_STEPS = HW_CFG["warmup_steps"]

    # 修复命令行参数被强制覆盖的 Bug
    if args.batch_size > 0:
        BATCH_SIZE = args.batch_size
    else:
        # 仅在使用默认配置时，应用 512 的硬上限
        BATCH_SIZE = min(BATCH_SIZE * 2, 512)

    MEMORY_SIZE = min(MEMORY_SIZE * 2, 100000)

    num_workers = max(2, hw.cpu_cores - 4)

    print("=" * 60)
    print("🚀 PARALLEL HEADLESS TRAINING v3.2")
    print(f"  Workers: {num_workers} / {hw.cpu_cores} cores")
    print(f"  Device     : {DEVICE}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Memory     : {MEMORY_SIZE}")
    print(f"  Hidden: {HIDDEN_DIM}")
    print("=" * 60)

    net = ReflexNet(hidden=HIDDEN_DIM).to(DEVICE)
    target_net = ReflexNet(hidden=HIDDEN_DIM).to(DEVICE)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    lr = args.lr if args.lr else LR
    optimizer = optim.AdamW(
        net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPS_DECAY, eta_min=LR_MIN)

    memory = LightPER(MEMORY_SIZE)
    pool = StrategyPool(max_size=8)
    amp_ctx = AMPContext(USE_AMP, DEVICE)

    start_ep, best_reward, stats = load_checkpoint(
        net, target_net, optimizer, scheduler,
        pool, HIDDEN_DIM, DEVICE)

    player_wins = stats.get("player_wins", 0)
    ai_wins = stats.get("ai_wins", 0)
    draws = stats.get("draws", 0)
    total_rounds = player_wins + ai_wins + draws
    ai_streak = 0
    best_streak = stats.get("best_streak", 0)
    global_step = start_ep * 100
    last_loss = 0.0
    current_lr = lr
    episode = start_ep

    save_interval = args.save_interval
    max_episodes = args.episodes
    WEIGHT_SYNC_INTERVAL = 20

    collector = ParallelCollector(num_workers, HW_CFG, net)
    collector.start()

    t_start = time.time()
    train_steps_total = 0
    last_report_time = time.time()

    print(f"\n  📊 Starting from episode {start_ep}")
    print(f"  💾 Save every {save_interval} episodes")
    if max_episodes > 0:
        print(f"  🎯 Target: {max_episodes} episodes")
    print(f"  ⏹️  Ctrl+C to stop & save\n")

    try:
        while True:
            if (max_episodes > 0
                    and episode >= start_ep + max_episodes):
                print(f"\n  ✅ Reached target "
                      f"{max_episodes} episodes")
                break

            transitions, results = (
                collector.collect_transitions(max_batches=200))

            for t in transitions:
                memory.push(t)

            for r in results:
                winner, total_reward, steps = r
                total_rounds += 1
                if winner == "ai":
                    ai_wins += 1
                    ai_streak += 1
                    best_streak = max(best_streak, ai_streak)
                elif winner == "player":
                    player_wins += 1
                    ai_streak = 0
                else:
                    draws += 1
                    ai_streak = 0

                wr = ai_wins / max(total_rounds, 1) * 100
                stats.setdefault("rewards", []).append(
                    total_reward)
                stats.setdefault("winrates", []).append(wr)
                stats.setdefault("losses", []).append(
                    last_loss)

                if total_reward > best_reward:
                    best_reward = total_reward
                    save_best(net, best_reward, episode, HIDDEN_DIM)

                episode += 1
                if episode > 0 and episode % 20 == 0:
                    rr = stats.get("rewards", [])
                    recent = rr[-20:] if len(rr) >= 20 else rr
                    if recent:
                        fitness = sum(recent) / len(recent)
                        pool.add(
                            f"gen{pool.generation}_ep{episode}",
                            net.state_dict(), fitness)

                if episode % save_interval == 0:
                    stats["player_wins"] = player_wins
                    stats["ai_wins"] = ai_wins
                    stats["draws"] = draws
                    stats["best_streak"] = best_streak
                    save_checkpoint(
                        net, target_net, optimizer, scheduler,
                        memory, stats, pool, episode,
                        best_reward, HIDDEN_DIM, DEVICE)

            #训练
            can_train = len(memory) >= max(
                BATCH_SIZE, WARMUP_STEPS)
            if can_train and transitions:
                train_iters = max(
                    4, len(transitions) // (BATCH_SIZE // 2))
                train_iters = min(train_iters, 100)

                for _ in range(train_iters):
                    if len(memory) < BATCH_SIZE:
                        break

                    per_beta = min(
                        1.0, 0.4 + global_step * 0.0001)
                    batch, idx, isw = memory.sample(
                        BATCH_SIZE, per_beta)

                    bs = torch.tensor(
                        np.array([t[0] for t in batch]),
                        dtype=torch.float32, device=DEVICE)
                    ba = torch.tensor(
                        [t[1] for t in batch],
                        dtype=torch.long,
                        device=DEVICE).unsqueeze(-1)
                    br = torch.tensor(
                        [t[2] for t in batch],
                        dtype=torch.float32,
                        device=DEVICE).unsqueeze(-1)
                    bs2 = torch.tensor(
                        np.array([t[3] for t in batch]),
                        dtype=torch.float32, device=DEVICE)
                    bd = torch.tensor(
                        [t[4] for t in batch],
                        dtype=torch.float32,
                        device=DEVICE).unsqueeze(-1)

                    with amp_ctx.autocast():
                        with torch.no_grad():
                            best_a = net(bs2).argmax(
                                dim=-1, keepdim=True)
                            q_next = target_net(bs2).gather(
                                1, best_a)
                            target = (
                                br + GAMMA ** N_STEP
                                * q_next * (1 - bd))
                        q_current = net(bs).gather(1, ba)
                        td_error = (
                            (target - q_current)
                            .detach().squeeze().cpu().numpy())
                        loss = (
                            isw.unsqueeze(-1).to(DEVICE)
                            * (q_current - target) ** 2).mean()
                    
                    optimizer.zero_grad()
                    amp_ctx.scale_and_step(
                        loss, optimizer,
                        net.parameters(), 10.0)
                    memory.update_priorities(idx, td_error)
                    last_loss = loss.item()
                    train_steps_total += 1
                    global_step += 1

                    for tp, sp in zip(
                            target_net.parameters(),
                            net.parameters()):
                        tp.data.copy_(
                            TAU * sp.data
                            + (1 - TAU) * tp.data)
                    scheduler.step()
                    current_lr = (
                        optimizer.param_groups[0]["lr"])

            # 权重同步
            if (train_steps_total > 0
                    and train_steps_total
                    % WEIGHT_SYNC_INTERVAL == 0):
                collector.broadcast_weights(net)

            # 日志
            now = time.time()
            if now - last_report_time >= 5.0:
                elapsed = now - t_start
                eps_done = episode - start_ep
                eps_per_min = (
                    eps_done / (elapsed / 60)
                    if elapsed > 0 else 0)
                wr = ai_wins / max(total_rounds, 1) * 100
                epsilon = max(
                    EPS_END,
                    EPS_START - global_step / EPS_DECAY)
                qsize = collector.get_queue_size()
                rr = stats.get("rewards", [-1])[-50:]
                avg_r = sum(rr) / max(len(rr), 1)

                print(
                    f"EP {episode:6d} |"
                    f"WR:{wr:5.1f}% |"
                    f"R:{avg_r:6.1f} |"
                    f"ε:{epsilon:.3f} |"
                    f"L:{last_loss:.4f} |"
                    f"lr:{current_lr:.1e} |"
                    f"Stk:{ai_streak:3d}/{best_streak} |"
                    f"Mem:{len(memory):6d} |"
                    f"Q:{qsize:3d} |"
                    f"T:{train_steps_total:7d} |"
                    f"{eps_per_min:6.1f}ep/m |"
                    f"{elapsed/3600:.2f}h")
                last_report_time = now

            if not transitions and not results:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\n  ⏹️  Interrupted by user")

    collector.stop()
    stats["player_wins"] = player_wins
    stats["ai_wins"] = ai_wins
    stats["draws"] = draws
    stats["best_streak"] = best_streak
    save_checkpoint(
        net, target_net, optimizer, scheduler,
        memory, stats, pool, episode, best_reward,
        HIDDEN_DIM, DEVICE)

    elapsed = time.time() - t_start
    eps_done = episode - start_ep
    wr = ai_wins / max(total_rounds, 1) * 100
    print(f"\n{'=' * 60}")
    print(f"  📊 Final Report")
    print(f"{'=' * 60}")
    print(f"  Episodes: {eps_done}")
    print(f"  Time       : {elapsed/3600:.2f} hours")
    print(f"  Speed      : "
          f"{eps_done/(elapsed/60):.1f} ep/min")
    print(f"  WinRate    : {wr:.1f}%")
    print(f"  Best Streak: {best_streak}")
    print(f"  Best Reward: {best_reward:.1f}")
    print(f"  Train Steps: {train_steps_total}")
    print(f"{'=' * 60}\n")


# ╔════════════════════════════════════════════╗
# ║              GUI主循环                     ║
# ╚════════════════════════════════════════════╝

def main_gui():
    args = parse_args()
    hw = HardwareProfile(forced_device=args.device)
    DEVICE = hw.device
    HW_CFG = hw.config
    # 👇 同样在这里新增这一行
    HW_CFG["hidden_dim"] = 128
    HIDDEN_DIM = HW_CFG["hidden_dim"]
    BATCH_SIZE = HW_CFG["batch_size"]
    MEMORY_SIZE = HW_CFG["memory_size"]
    TRAIN_PER_FRAME = HW_CFG["train_steps_per_frame"]
    USE_AMP = HW_CFG["use_amp"]
    GRAD_ACCUM = HW_CFG["grad_accum"]
    MAX_FPS_TRAIN = HW_CFG["max_fps_train"]
    WARMUP_STEPS = HW_CFG["warmup_steps"]

    if args.batch_size > 0:
        BATCH_SIZE = args.batch_size

    import pygame
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"Grid Duel Arena v{VERSION}")
    clock = pygame.time.Clock()

    try:
        fonts = {
            "sm": pygame.font.SysFont("consolas", 11),
            "md": pygame.font.SysFont("consolas", 13, bold=True),
            "lg": pygame.font.SysFont("consolas", 17, bold=True),
            "xl": pygame.font.SysFont("consolas", 30, bold=True),
        }
    except Exception:
        fonts = {
            "sm": pygame.font.SysFont(None, 14),
            "md": pygame.font.SysFont(None, 16),
            "lg": pygame.font.SysFont(None, 20),
            "xl": pygame.font.SysFont(None, 34),
        }

    net = ReflexNet(hidden=HIDDEN_DIM).to(DEVICE)
    target_net = ReflexNet(hidden=HIDDEN_DIM).to(DEVICE)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(
        net.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPS_DECAY, eta_min=LR_MIN)

    memory = LightPER(MEMORY_SIZE)
    nstep = NStepBuffer(N_STEP, GAMMA)
    pool = StrategyPool(max_size=8)

    opponent_net = ReflexNet(hidden=HIDDEN_DIM).to(DEVICE)
    opponent_net.eval()

    amp_ctx = AMPContext(USE_AMP, DEVICE)

    start_ep, best_reward, stats = load_checkpoint(
        net, target_net, optimizer, scheduler,
        pool, HIDDEN_DIM, DEVICE)

    renderer = Renderer(screen, clock, fonts, hw.tier)

    for v in stats.get("winrates", [])[-150:]:
        renderer.chart_winrate.add(v)
    for v in stats.get("rewards", [])[-150:]:
        renderer.chart_reward.add(v)
    for v in stats.get("losses", [])[-150:]:
        renderer.chart_loss.add(v)

    dev_str = str(DEVICE)
    if hw.has_cuda:
        dev_str += (f" ({hw.gpu_name[:25]}, " f"{hw.vram_mb}MB)")

    mode = "PvAI"
    speed = 1
    paused = False
    show_help = False
    global_step = start_ep * 100
    last_loss = 0.0
    gate_val = 0.5
    player_wins = stats.get("player_wins", 0)
    ai_wins = stats.get("ai_wins", 0)
    draws = stats.get("draws", 0)
    total_rounds = player_wins + ai_wins + draws
    player_action_queue = 5
    current_lr = LR
    ai_streak = 0
    best_streak = stats.get("best_streak", 0)
    grad_accum_counter = 0

    print(f"\n  🏟️  Grid Duel Arena v{VERSION}")
    print(f"     Device: {DEVICE} | Tier: {hw.tier}")
    print(f"     [1] PvAI  [2] SelfPlay  "
          f"[3] FastTrain\n")

    episode = start_ep

    running = True
    while running:
        world = DuelArena(seed=random.randint(0, 2**31))
        obs = world.reset()
        total_reward = 0.0
        nstep.reset()
        show_result = False
        result_timer = 0

        epsilon = max(
            EPS_END, EPS_START - global_step / EPS_DECAY)

        is_eval = (episode % EVAL_INTERVAL == 0
                   and episode > 0
                   and mode in ("SelfPlay", "Train"))
        eval_epsilon = 0.0 if is_eval else epsilon

        use_pool_opp = False
        opp_name = "Rule"
        if mode in ("SelfPlay", "Train"):
            pool_prob = (
                min(0.5, episode / 1000.0)
                if pool.pool else 0.0)
            if random.random() < pool_prob:
                opp = pool.sample_opponent()
                if opp is not None:
                    try:
                        migrate_weights(
                            opponent_net, opp[1], "opp")
                        use_pool_opp = True
                        opp_name = opp[0]
                    except Exception:
                        use_pool_opp = False

        round_running = True
        while round_running and running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                    break
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                        break
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
                        speed = max(speed, 10)
                    elif ev.key == pygame.K_TAB:
                        show_help = not show_help
                    elif (ev.key == pygame.K_n
                          and show_result):
                        round_running = False
                        continue
                    elif (ev.key == pygame.K_s
                          and (pygame.key.get_mods()
                               & pygame.KMOD_CTRL)
                          and not show_result):
                        stats["player_wins"] = player_wins
                        stats["ai_wins"] = ai_wins
                        stats["draws"] = draws
                        stats["best_streak"] = best_streak
                        save_checkpoint(
                            net, target_net, optimizer,
                            scheduler, memory, stats, pool,
                            episode, best_reward,
                            HIDDEN_DIM, DEVICE)
                    if (mode == "PvAI"
                            and not show_result):
                        if ev.key == pygame.K_w:
                            player_action_queue = 0
                        elif (ev.key == pygame.K_s
                              and not (pygame.key.get_mods()
                                       & pygame.KMOD_CTRL)):
                            player_action_queue = 1
                        elif ev.key == pygame.K_a:
                            player_action_queue = 2
                        elif ev.key == pygame.K_d:
                            player_action_queue = 3
                        elif ev.key == pygame.K_j:
                            player_action_queue = 4
                        elif ev.key == pygame.K_k:
                            player_action_queue = 5

            if not running:
                break

            if (mode == "PvAI" and not show_result
                    and not paused):
                keys = pygame.key.get_pressed()
                mods = pygame.key.get_mods()
                ctrl = bool(mods & pygame.KMOD_CTRL)
                if keys[pygame.K_w]:
                    player_action_queue = 0
                elif keys[pygame.K_s] and not ctrl:
                    player_action_queue = 1
                elif keys[pygame.K_a]:
                    player_action_queue = 2
                elif keys[pygame.K_d]:
                    player_action_queue = 3

            fps_val = clock.get_fps()

            if paused and not show_result:
                renderer.draw_arena(world)
                renderer.draw_panel(
                    world, episode, eval_epsilon,
                    last_loss, mode, speed, ai_wins,
                    player_wins, total_rounds, gate_val,
                    pool.generation, fps_val, current_lr,
                    ai_streak, best_streak, is_eval)
                renderer.draw_bottom(
                    mode, eval_epsilon, global_step,
                    dev_str)
                ptxt = fonts["lg"].render(
                    "PAUSED", True, C_WARN)
                screen.blit(
                    ptxt,
                    (ARENA_W // 2 - ptxt.get_width() // 2, ARENA_H // 2 - ptxt.get_height() // 2))
                if show_help:
                    renderer.draw_help_overlay(
                        dev_str, HIDDEN_DIM, BATCH_SIZE,
                        MEMORY_SIZE, USE_AMP, GRAD_ACCUM, WARMUP_STEPS)
                pygame.display.flip()
                clock.tick(15)
                continue

            if show_result:
                renderer.draw_arena(world)
                renderer.draw_panel(
                    world, episode, eval_epsilon,
                    last_loss, mode, speed, ai_wins,
                    player_wins, total_rounds, gate_val,
                    pool.generation, fps_val, current_lr,
                    ai_streak, best_streak, is_eval)
                renderer.draw_bottom(
                    mode, eval_epsilon, global_step,
                    dev_str)
                renderer.draw_round_result(
                    world.winner, player_wins,
                    ai_wins, total_rounds)
                if show_help:
                    renderer.draw_help_overlay(
                        dev_str, HIDDEN_DIM, BATCH_SIZE,
                        MEMORY_SIZE, USE_AMP, GRAD_ACCUM,
                        WARMUP_STEPS)
                pygame.display.flip()
                clock.tick(15)
                if mode in ("SelfPlay", "Train"):
                    result_timer += 1
                    limit = 2 if mode == "Train" else 12
                    if result_timer > limit:
                        round_running = False
                continue

            if mode == "PvAI":
                p_act = player_action_queue
                player_action_queue = 5
            elif use_pool_opp:
                st = world.get_state(for_ai=False)
                st_t = torch.tensor(
                    st, dtype=torch.float32,
                    device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    q = opponent_net(st_t)
                p_act = q.argmax(dim=-1).item()
            else:
                p_act = rule_based_player(world)

            obs_t = torch.tensor(
                obs, dtype=torch.float32,
                device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q_values = net(obs_t)
                gate_val = net.get_gate_value(obs_t)

            q_np = q_values.squeeze(0).cpu().numpy()
            renderer.qbar.update(q_np)

            if random.random() < eval_epsilon:
                ai_act = random.randint(0, ACTION_DIM - 1)
            else:
                ai_act = q_values.argmax(dim=-1).item()

            prev_ai_hp = world.ai.hp
            prev_player_hp = world.player.hp
            obs2, reward, done = world.step(p_act, ai_act)
            total_reward += reward
            global_step += 1

            for exp in world.new_explosions:
                renderer.add_explosion_particles(
                    exp.x, exp.y)
            if world.ai.hp < prev_ai_hp:
                renderer.add_hit_particles(
                    world.ai.x, world.ai.y, C_AI)
            if world.player.hp < prev_player_hp:
                renderer.add_hit_particles(
                    world.player.x, world.player.y,
                    C_PLAYER)

            if not is_eval:
                nstep.push(
                    (obs, ai_act, reward, obs2, float(done)))
                nt = nstep.get()
                if nt:
                    memory.push(nt)
            obs = obs2

            can_train = (
                len(memory) >= max(BATCH_SIZE, WARMUP_STEPS)
                and not is_eval)
            train_iters = (TRAIN_PER_FRAME
                if (not done and can_train)
                else (1 if can_train else 0))

            for _ in range(train_iters):
                if len(memory) < BATCH_SIZE:
                    break

                per_beta = min(
                    1.0, 0.4 + global_step * 0.0001)
                batch, idx, isw = memory.sample(
                    BATCH_SIZE, per_beta)

                bs = torch.tensor(
                    np.array([t[0] for t in batch]),
                    dtype=torch.float32, device=DEVICE)
                ba = torch.tensor(
                    [t[1] for t in batch],
                    dtype=torch.long,
                    device=DEVICE).unsqueeze(-1)
                br = torch.tensor(
                    [t[2] for t in batch],
                    dtype=torch.float32,
                    device=DEVICE).unsqueeze(-1)
                bs2 = torch.tensor(
                    np.array([t[3] for t in batch]),
                    dtype=torch.float32, device=DEVICE)
                bd = torch.tensor(
                    [t[4] for t in batch],
                    dtype=torch.float32,
                    device=DEVICE).unsqueeze(-1)

                with amp_ctx.autocast():
                    with torch.no_grad():
                        best_a = net(bs2).argmax(
                            dim=-1, keepdim=True)
                        q_next = target_net(bs2).gather(
                            1, best_a)
                        target = (
                            br + GAMMA ** N_STEP
                            * q_next * (1 - bd))
                    q_current = net(bs).gather(1, ba)
                    td_error = (
                        (target - q_current)
                        .detach().squeeze().cpu().numpy())
                    loss = (
                        isw.unsqueeze(-1).to(DEVICE)
                        * (q_current - target) ** 2).mean()

                if GRAD_ACCUM > 1:
                    loss = loss / GRAD_ACCUM
                    grad_accum_counter += 1
                    if grad_accum_counter == 1:
                        optimizer.zero_grad()
                    if grad_accum_counter >= GRAD_ACCUM:
                        amp_ctx.scale_and_step(
                            loss, optimizer,
                            net.parameters(), 10.0)
                        grad_accum_counter = 0
                    else:
                        loss.backward()
                else:
                    optimizer.zero_grad()
                    amp_ctx.scale_and_step(
                        loss, optimizer,
                        net.parameters(), 10.0)

                memory.update_priorities(idx, td_error)
                last_loss = loss.item() * (GRAD_ACCUM if GRAD_ACCUM > 1 else 1)

                for tp, sp in zip(
                        target_net.parameters(),
                        net.parameters()):
                    tp.data.copy_(
                        TAU * sp.data
                        + (1 - TAU) * tp.data)
                scheduler.step()
                current_lr = (
                    optimizer.param_groups[0]["lr"])

            do_render = True
            if mode == "Train" and global_step % 4 != 0:
                do_render = False

            if do_render:
                renderer.draw_arena(world)
                renderer.draw_panel(
                    world, episode, eval_epsilon,
                    last_loss, mode, speed, ai_wins,
                    player_wins, total_rounds, gate_val,
                    pool.generation, fps_val, current_lr,
                    ai_streak, best_streak, is_eval)
                renderer.draw_bottom(
                    mode, eval_epsilon, global_step,
                    dev_str)
                if show_help:
                    renderer.draw_help_overlay(
                        dev_str, HIDDEN_DIM, BATCH_SIZE,
                        MEMORY_SIZE, USE_AMP, GRAD_ACCUM,
                        WARMUP_STEPS)
                pygame.display.flip()

            fps = BASE_FPS * speed
            if mode == "Train":
                fps = max(fps, MAX_FPS_TRAIN)
            clock.tick(fps)

            if done:
                if not is_eval:
                    for t in nstep.flush():
                        memory.push(t)

                show_result = True
                result_timer = 0
                total_rounds += 1

                if world.winner == "ai":
                    ai_wins += 1
                    ai_streak += 1
                    best_streak = max(
                        best_streak, ai_streak)
                elif world.winner == "player":
                    player_wins += 1
                    ai_streak = 0
                else:
                    draws += 1
                    ai_streak = 0

                wr = ai_wins / max(total_rounds, 1) * 100
                stats.setdefault("rewards", []).append(
                    total_reward)
                stats.setdefault("winrates", []).append(wr)
                stats.setdefault("losses", []).append(
                    last_loss)

                renderer.chart_winrate.add(wr)
                renderer.chart_reward.add(total_reward)
                renderer.chart_eps.add(eval_epsilon)
                renderer.chart_loss.add(last_loss)

                winner_str = {
                    "ai": "AI WIN", "player": "P WIN",
                    "draw": "DRAW"
                }.get(world.winner, "?")
                eval_tag = " [EVAL]" if is_eval else ""

                print(
                    f"EP {episode:5d}|"
                    f"{winner_str:>6s}{eval_tag}|"
                    f"R:{total_reward:7.1f}|"
                    f"AI:{ai_wins} P:{player_wins}|"
                    f"WR:{wr:5.1f}%|"
                    f"e:{eval_epsilon:.3f}|"
                    f"G:{gate_val:.2f}|"
                    f"lr:{current_lr:.1e}|"
                    f"Stk:{ai_streak}|"
                    f"Opp:{opp_name}|"
                    f"Mem:{len(memory)}")

                if (episode > 0
                        and episode % 20 == 0
                        and not is_eval):
                    rr = stats["rewards"]
                    recent = (rr[-20:]
                              if len(rr) >= 20 else rr)
                    fitness = (
                        sum(recent) / max(len(recent), 1))
                    pool.add(
                        f"gen{pool.generation}_ep{episode}",
                        net.state_dict(), fitness)

                if total_reward > best_reward:
                    best_reward = total_reward
                    save_best(net, best_reward, episode,
                              HIDDEN_DIM)

                if (episode + 1) % 50 == 0:
                    stats["player_wins"] = player_wins
                    stats["ai_wins"] = ai_wins
                    stats["draws"] = draws
                    stats["best_streak"] = best_streak
                    save_checkpoint(
                        net, target_net, optimizer,
                        scheduler, memory, stats, pool,
                        episode + 1, best_reward,
                        HIDDEN_DIM, DEVICE)

                episode += 1

    stats["player_wins"] = player_wins
    stats["ai_wins"] = ai_wins
    stats["draws"] = draws
    stats["best_streak"] = best_streak
    save_checkpoint(
        net, target_net, optimizer, scheduler,
        memory, stats, pool, episode, best_reward,
        HIDDEN_DIM, DEVICE)
    pygame.quit()
    print("\n👋 Game saved. Goodbye!")


# ╔════════════════════════════════════════════╗
# ║      ★★★入口：只在主进程执行 ★★★           ║
# ╚════════════════════════════════════════════╝

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = parse_args()
    if args.headless:
        headless = True
    elif args.gui:
        headless = False
    else:
        headless = not detect_display()

    if headless:
        print("\n  🖥️  Headless mode — pure training")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        main_headless()
    else:
        print("\n  🎮  GUI mode — interactive")
        main_gui()