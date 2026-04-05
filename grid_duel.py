#!/usr/bin/env python3
"""
Grid Duel Arena v2.0 — Biomimetic Lightweight AI Combat Trainer
===============================================================
炸弹对战竞技场：玩家 vs 自我进化AI

v2.0 更新:
  ✦ 自动检测硬件 (GPU/CPU/VRAM) 自适应训练方案
  ✦ 完全兼容 v1.0 模型存档
  ✦ 修复全部已知bug
  ✦ 改进操控手感与UI体验
  ✦ 自适应批量/网络宽度/显存占用

pip install torch pygame numpy
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
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

# ╔════════════════════════════════════════════╗
# ║     硬件自动检测 & 自适应配置              ║
# ╚════════════════════════════════════════════╝

class HardwareProfile:
    """自动检测硬件能力，生成最优训练配置"""

    def __init__(self):
        self.has_cuda = torch.cuda.is_available()
        self.gpu_name = ""
        self.vram_mb = 0
        self.cpu_cores = os.cpu_count() or 2
        self.ram_mb = self._get_ram()
        self.tier = "cpu_low"
        self.device = torch.device("cpu")
        
        if self.has_cuda:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
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
                c_ulonglong = ctypes.c_ulonglong
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", c_ulonglong),
                        ("ullAvailPhys", c_ulonglong),
                        ("ullTotalPageFile", c_ulonglong),
                        ("ullAvailPageFile", c_ulonglong),
                        ("ullTotalVirtual", c_ulonglong),
                        ("ullAvailVirtual", c_ulonglong),
                        ("sullAvailExtendedVirtual", c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return stat.ullTotalPhys // (1024 * 1024)
            elif sys.platform == "darwin":
                import subprocess
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
                return int(out.strip()) // (1024 * 1024)
        except Exception:
            pass
        return 4096  # 默认4GB

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
            self.device = torch.device("cpu")

    def _classify_cpu(self):
        if self.ram_mb >= 8000 and self.cpu_cores >= 4:
            self.tier = "cpu_high"
        elif self.ram_mb >= 4000:
            self.tier = "cpu_mid"
        else:
            self.tier = "cpu_low"

    def _build_config(self):
        """根据硬件层级生成训练参数"""
        profiles = {
            "gpu_large": {
                "hidden_dim": 192,
                "batch_size": 256,
                "memory_size": 50000,
                "train_steps_per_frame": 2,
                "max_fps_train": 300,
                "use_amp": True,
                "grad_accum": 1,
            },
            "gpu_medium": {
                "hidden_dim": 160,
                "batch_size": 128,
                "memory_size": 30000,
                "train_steps_per_frame": 2,
                "max_fps_train": 240,
                "use_amp": True,
                "grad_accum": 1,
            },
            "gpu_small": {
                "hidden_dim": 128,
                "batch_size": 96,
                "memory_size": 20000,
                "train_steps_per_frame": 1,
                "max_fps_train": 180,
                "use_amp": True,
                "grad_accum": 1,
            },
            "gpu_tiny": {
                "hidden_dim": 96,
                "batch_size": 48,
                "memory_size": 12000,
                "train_steps_per_frame": 1,
                "max_fps_train": 120,
                "use_amp": False,
                "grad_accum": 2,
            },
            "cpu_high": {
                "hidden_dim": 128,
                "batch_size": 64,
                "memory_size": 15000,
                "train_steps_per_frame": 1,
                "max_fps_train": 120,
                "use_amp": False,
                "grad_accum": 1,
            },
            "cpu_mid": {
                "hidden_dim": 96,
                "batch_size": 48,
                "memory_size": 10000,
                "train_steps_per_frame": 1,
                "max_fps_train": 90,
                "use_amp": False,
                "grad_accum": 2,
            },
            "cpu_low": {
                "hidden_dim": 64,
                "batch_size": 32,
                "memory_size": 8000,
                "train_steps_per_frame": 1,
                "max_fps_train": 60,
                "use_amp": False,
                "grad_accum": 2,
            },
            "cpu_fallback": {
                "hidden_dim": 64,
                "batch_size": 32,
                "memory_size": 8000,
                "train_steps_per_frame": 1,
                "max_fps_train": 60,
                "use_amp": False,
                "grad_accum": 2,
            },
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
        c = self.config
        print(f"  Hidden : {c['hidden_dim']}")
        print(f"  Batch  : {c['batch_size']}")
        print(f"  Memory : {c['memory_size']}")
        print(f"  AMP    : {c['use_amp']}")
        print(f"  GradAcc: {c['grad_accum']}")
        print(f"  TrainFPS: {c['max_fps_train']}")
        print("=" * 56 + "\n")


# ╔════════════════════════════════════════════╗
# ║              全局常量                ║
# ╚════════════════════════════════════════════╝

VERSION = "2.0"
PREV_VERSIONS = ["1.0"]

HW = HardwareProfile()
DEVICE = HW.device
HW_CFG = HW.config

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
HIDDEN_DIM = HW_CFG["hidden_dim"]
BATCH_SIZE = HW_CFG["batch_size"]
MEMORY_SIZE = HW_CFG["memory_size"]
TRAIN_PER_FRAME = HW_CFG["train_steps_per_frame"]
USE_AMP = HW_CFG["use_amp"]
GRAD_ACCUM = HW_CFG["grad_accum"]
MAX_FPS_TRAIN = HW_CFG["max_fps_train"]

GAMMA = 0.97
LR = 5e-4
TAU = 0.01
EPS_START = 1.0
EPS_END = 0.03
EPS_DECAY = 5000
N_STEP = 3

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
    0: (0, -1),
    1: (0, 1),
    2: (-1, 0),
    3: (1, 0),
    4: (0, 0),
    5: (0, 0),
}
DIR_NAMES = ["Up", "Dn", "Lt", "Rt", "Bomb", "Stay"]

# ╔════════════════════════════════════════════╗
# ║                颜色主题                    ║
# ╚════════════════════════════════════════════╝

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
C_EXPLODE = [
    (255, 255, 200),
    (255, 200, 50),
    (255, 120, 20),
    (200, 50, 0),
]
C_POWERUP = (100, 255, 200)
C_SHIELD = (80, 160, 255)

C_CHART_WIN = (80, 200, 255)
C_CHART_REW = (255, 175, 55)
C_CHART_EPS = (200, 100, 255)


# ╔════════════════════════════════════════════╗
# ║              地图生成器                    ║
# ╚════════════════════════════════════════════╝

EMPTY = 0
WALL = 1
BRICK = 2


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
# ║            游戏对象定义                    ║
# ╚════════════════════════════════════════════╝

class Bomb:
    __slots__ = ["x", "y", "timer", "owner", "power"]

    def __init__(self, x, y, owner, power=BOMB_RANGE):
        self.x = x
        self.y = y
        self.timer = BOMB_TIMER
        self.owner = owner
        self.power = power


class Explosion:
    __slots__ = ["x", "y", "timer"]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.timer = 4


class PowerUp:
    __slots__ = ["x", "y", "kind"]

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
        self.speed = 1
        self.invincible = 0
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


# ╔════════════════════════════════════════════╗
# ║            DuelArena 游戏世界              ║
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
        if fighter.bomb_cooldown > 0:
            fighter.bomb_cooldown -= 1
        if fighter.invincible > 0:
            fighter.invincible -= 1

        if action == 4:
            if (fighter.bomb_cooldown <= 0 and fighter.active_bombs < fighter.max_bombs):
                has_bomb = any(
                    b.x == fighter.x and b.y == fighter.y
                    for b in self.bombs
                )
                if not has_bomb:
                    self.bombs.append(
                        Bomb(fighter.x, fighter.y, fighter, fighter.bomb_power)
                    )
                    fighter.active_bombs += 1
                    fighter.bomb_cooldown = BOMB_COOLDOWN
        elif action < 4:
            dx, dy = DIR_MAP[action]
            nx, ny = fighter.x + dx, fighter.y + dy
            if self._can_move(nx, ny):
                other = self.ai if not fighter.is_ai else self.player
                if not (nx == other.x and ny == other.y):
                    fighter.x = nx
                    fighter.y = ny

    def _explode_bomb(self, bomb):
        cells = [(bomb.x, bomb.y)]
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in dirs:
            for r in range(1, bomb.power + 1):
                ex = bomb.x + dx * r
                ey = bomb.y + dy * r
                if ex < 0 or ex >= ARENA_COLS or ey < 0 or ey >= ARENA_ROWS:
                    break
                if self.grid[ey][ex] == WALL:
                    break
                cells.append((ex, ey))
                if self.grid[ey][ex] == BRICK:
                    self.grid[ey][ex] = EMPTY
                    if self.rng.random() < 0.25:
                        kind = self.rng.randint(0, 2)
                        self.powerups.append(PowerUp(ex, ey, kind))
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
                        fighter.bomb_power = min(fighter.bomb_power + 1, 5)
                    elif pu.kind == 1:
                        fighter.speed = min(fighter.speed + 1, 3)
                    elif pu.kind == 2:
                        fighter.max_bombs = min(fighter.max_bombs + 1, 3)
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

        # 炸弹倒计时 + 连锁爆炸
        exploded_cells = []
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(self.bombs):
                self.bombs[i].timer -= 1
                if self.bombs[i].timer <= 0:
                    cells = self._explode_bomb(self.bombs[i])
                    exploded_cells.extend(cells)
                    for cx, cy in cells:
                        exp = Explosion(cx, cy)
                        self.explosions.append(exp)
                        self.new_explosions.append(exp)
                    # 链爆检查
                    for j in range(len(self.bombs)):
                        if j != i and self.bombs[j].timer > 0:
                            for cx, cy in cells:
                                if (self.bombs[j].x == cx
                                        and self.bombs[j].y == cy):
                                    self.bombs[j].timer = 0
                                    changed = True
                    self.bombs[i].owner.active_bombs = max(
                        0, self.bombs[i].owner.active_bombs - 1)
                    self.bombs.pop(i)
                    changed = True
                    break
                else:
                    i += 1

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
            ai_reward = 0.0
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
        reward = 0.01

        dist = abs(self.ai.x - self.player.x) + abs(self.ai.y - self.player.y)
        if dist <= 3:
            reward += 0.05
        elif dist >= 8:
            reward -= 0.02

        for exp in self.explosions:
            if abs(self.ai.x - exp.x) + abs(self.ai.y - exp.y) <= 1:
                reward -= 0.3

        for b in self.bombs:
            bd = abs(self.ai.x - b.x) + abs(self.ai.y - b.y)
            if bd <= 1 and b.timer <= 3:
                reward -= 0.15

        hp_diff = self.ai.hp - self.player.hp
        reward += hp_diff * 0.05

        return reward

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
                if self.grid[y][x] == EMPTY and (x, y) not in occupied:
                    empties.append((x, y))

        if empties and len(self.powerups) < 4:
            x, y = self.rng.choice(empties)
            kind = self.rng.randint(0, 2)
            self.powerups.append(PowerUp(x, y, kind))

    def get_state(self, for_ai=True):
        me = self.ai if for_ai else self.player
        enemy = self.player if for_ai else self.ai

        features = []

        # 自身(6)
        features.append(me.x / ARENA_COLS)
        features.append(me.y / ARENA_ROWS)
        features.append(me.hp / me.max_hp)
        features.append(me.bomb_cooldown / BOMB_COOLDOWN)
        features.append(me.active_bombs / max(me.max_bombs, 1))
        features.append(me.invincible / 6.0)

        # 敌方 (5)
        features.append((enemy.x - me.x) / ARENA_COLS)
        features.append((enemy.y - me.y) / ARENA_ROWS)
        features.append(enemy.hp / enemy.max_hp)
        dist = abs(enemy.x - me.x) + abs(enemy.y - me.y)
        features.append(dist / (ARENA_COLS + ARENA_ROWS))
        angle = math.atan2(enemy.y - me.y, enemy.x - me.x)
        features.append(angle / math.pi)

        # 四向近场感知(12)
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in dirs:
            wall_dist = 0
            for r in range(1, max(ARENA_COLS, ARENA_ROWS)):
                cx = me.x + dx * r
                cy = me.y + dy * r
                if (cx < 0 or cx >= ARENA_COLS
                        or cy < 0 or cy >= ARENA_ROWS):
                    break
                if self.grid[cy][cx] in (WALL, BRICK):
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

        # 炸弹全局(6)
        my_bomb_count = sum(1 for b in self.bombs if b.owner is me)
        enemy_bomb_count = sum(1 for b in self.bombs if b.owner is enemy)
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
                    or (b.y == me.y and abs(b.x - me.x) <= b.power)):
                in_danger = max(in_danger, 1.0 - b.timer / BOMB_TIMER)
        features.append(in_danger)

        stuck = 1.0
        for ddx, ddy in dirs:
            if self._can_move(me.x + ddx, me.y + ddy):
                stuck = 0.0
                break
        features.append(stuck)

        # 道具 (3)
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

        # 局势 (4)
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
                    ex = b.x + dx * r
                    ey = b.y + dy * r
                    if 0 <= ex < ARENA_COLS and 0 <= ey < ARENA_ROWS:
                        if self.grid[ey][ex] == WALL:
                            break
                        dmap[ey][ex] = max(dmap[ey][ex], urgency * 0.7)
                    else:
                        break
        for exp in self.explosions:
            if 0 <= exp.y < ARENA_ROWS and 0 <= exp.x < ARENA_COLS:
                dmap[exp.y][exp.x] = 1.0
        return dmap


# ╔════════════════════════════════════════════╗
# ║      神经网络 (蟑螂反射弧,兼容v1.0)        ║
# ╚════════════════════════════════════════════╝

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ReflexNet(nn.Module):
    """蟑螂反射弧双通道网络。
    hidden_dim 可变，支持不同硬件档位。
    兼容 v1.0 (hidden=128) 权重加载。
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden=None):
        super().__init__()
        if hidden is None:
            hidden = HIDDEN_DIM
        self.hidden = hidden
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 快速反射通道 (近场18维→ action_dim)
        self.fast_net = nn.Sequential(
            nn.Linear(18, 48),
            Swish(),
            nn.Linear(48, 24),
            Swish(),
            nn.Linear(24, action_dim),
        )

        # 慢速策略通道 (全部state_dim → action_dim)
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

        # Dueling Head
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

        # 门控
        self.gate = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.shape[-1] >= 29:
            fast_feat = x[:, 11:29]
        else:
            fast_feat = x[:, :18]

        fast_q = self.fast_net(fast_feat)
        slow_q = self.slow_net(x)

        g = self.gate(x)
        combined = torch.cat([fast_q * g, slow_q * (1 - g)], dim=-1)

        v = self.value_head(combined)
        a = self.advantage_head(combined)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q

    def get_gate_value(self, x):
        with torch.no_grad():
            g = self.gate(x)
            return g.mean().item()


# ╔════════════════════════════════════════════╗
# ║      经验回放 (轻量PER + NStep)          ║
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
        return samples, idx, torch.tensor(w, dtype=torch.float32,
                                          device=DEVICE)

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
        s0 = self.buffer[0][0]
        a0 = self.buffer[0][1]
        r = sum(self.gamma ** i * self.buffer[i][2]
                for i in range(self.n))
        return (s0, a0, r, self.buffer[-1][3], self.buffer[-1][4])

    def flush(self):
        results = []
        while self.buffer:
            s0 = self.buffer[0][0]
            a0 = self.buffer[0][1]
            r = sum(self.gamma ** i * self.buffer[i][2]
                    for i in range(len(self.buffer)))
            results.append(
                (s0, a0, r, self.buffer[-1][3], self.buffer[-1][4])
            )
            self.buffer.popleft()
        return results

    def reset(self):
        self.buffer.clear()


# ╔════════════════════════════════════════════╗
# ║      策略池 (免疫克隆选择)               ║
# ╚════════════════════════════════════════════╝

class StrategyPool:
    def __init__(self, max_size=8):
        self.max_size = max_size
        self.pool = []
        self.generation = 0

    def add(self, name, state_dict, fitness):
        self.pool.append((name, copy.deepcopy(state_dict), fitness))
        self.generation += 1
        if len(self.pool) > self.max_size:
            self.pool.sort(key=lambda x: x[2], reverse=True)
            self.pool = self.pool[:self.max_size]

    def sample_opponent(self):
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
# ║              UI 小组件                   ║
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
        pygame.draw.rect(surf, (10, 10, 22),
                         (self.x, self.y, self.w, self.h))
        pygame.draw.rect(surf, (40, 40, 60),
                         (self.x, self.y, self.w, self.h), 1)
        surf.blit(font.render(self.title, True, C_DIM), (self.x + 4, self.y + 2))
        if len(self.data) < 2:
            return
        dl = list(self.data)
        mn = min(dl)
        mx = max(dl)
        rng = mx - mn if mx != mn else 1.0
        cy = self.y + 15
        ch = self.h - 18

        surf.blit(font.render(f"{dl[-1]:.1f}", True, self.color),
                  (self.x + self.w - 48, self.y + 2))

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
            dim_c = tuple(v // 3 for v in self.color)
            pygame.draw.lines(surf, dim_c, False, pts, 1)
        if len(apts) >= 2:
            pygame.draw.lines(surf, self.color, False, apts, 2)


class Particle:
    __slots__ = ["x", "y", "vx", "vy", "life", "max_life", "color", "size"]

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
        a = max(0.0, self.life / self.max_life)
        r = int(self.size * a)
        if r > 0:
            c = tuple(min(255, int(v * a)) for v in self.color)
            pygame.draw.circle(surf, c, (int(self.x), int(self.y)), r)


# ╔════════════════════════════════════════════╗
# ║              主渲染器                    ║
# ╚════════════════════════════════════════════╝

class Renderer:
    def __init__(self, screen, clock, fonts):
        self.screen = screen
        self.clock = clock
        self.fonts = fonts
        self.particles = []
        self.pulse = 0.0
        px = ARENA_W + 10
        cw = PANEL_W - 20
        self.chart_winrate = MiniChart(px, 10, cw, 70,
                                       "AI WinRate%", C_CHART_WIN)
        self.chart_reward = MiniChart(px, 90, cw, 70, "AI Reward", C_CHART_REW)
        self.chart_eps = MiniChart(px, 170, cw, 70,
                                   "Epsilon", C_CHART_EPS)

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

        self.screen.fill(C_BG)

        dmap = world.get_danger_map()

        for y in range(ARENA_ROWS):
            for x in range(ARENA_COLS):
                rx = x * CELL
                ry = y * CELL
                cell = world.grid[y][x]
                if cell == WALL:
                    pygame.draw.rect(self.screen, C_WALL,
                                     (rx, ry, CELL, CELL))
                    pygame.draw.rect(self.screen, C_WALL_L,
                                     (rx + 1, ry + 1, CELL - 2, CELL - 2),
                                     1)
                elif cell == BRICK:
                    pygame.draw.rect(self.screen, C_BRICK,
                                     (rx, ry, CELL, CELL))
                    pygame.draw.rect(self.screen, C_BRICK_D,
                                     (rx + 2, ry + 2, CELL - 4, CELL - 4))
                    pygame.draw.line(self.screen, C_BRICK_D,
                                     (rx, ry + CELL // 2),
                                     (rx + CELL, ry + CELL // 2), 1)
                    pygame.draw.line(self.screen, C_BRICK_D,
                                     (rx + CELL // 2, ry),
                                     (rx + CELL // 2, ry + CELL), 1)
                else:
                    pygame.draw.rect(self.screen, C_FLOOR,
                                     (rx, ry, CELL, CELL))
                    d = dmap[y][x]
                    if d > 0.01:
                        ds = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                        ds.fill((255, 50, 0, int(d * 100)))
                        self.screen.blit(ds, (rx, ry))
                pygame.draw.rect(self.screen, C_GRID,
                                 (rx, ry, CELL, CELL), 1)

        # 道具
        pu_icons = ["R", "S", "B"]
        pu_colors = [C_POWERUP, C_WARN, C_SHIELD]
        for pu in world.powerups:
            px_ = pu.x * CELL + CELL // 2
            py_ = pu.y * CELL + CELL // 2
            r = int(CELL * 0.3 + math.sin(self.pulse * 2) * 2)
            c = pu_colors[pu.kind % 3]
            pygame.draw.circle(self.screen, c, (px_, py_), max(r, 4))
            txt = self.fonts["sm"].render(pu_icons[pu.kind % 3],
                                          True, (0, 0, 0))
            self.screen.blit(txt, (px_ - txt.get_width() // 2,
                                   py_ - txt.get_height() // 2))

        # 炸弹
        for b in world.bombs:
            bx = b.x * CELL + CELL // 2
            by = b.y * CELL + CELL // 2
            urgency = 1.0 - b.timer / BOMB_TIMER
            r = int(CELL * 0.35 + urgency * 4)
            flash = b.timer <= 3 and b.timer % 2 == 0
            bc = C_BOMB_FUSE if flash else C_BOMB
            pygame.draw.circle(self.screen, bc, (bx, by), max(r, 4))
            pygame.draw.line(self.screen, C_BOMB_FUSE,
                             (bx, by - r), (bx + 3, by - r - 5), 2)
            txt = self.fonts["sm"].render(str(b.timer), True, (0, 0, 0))
            self.screen.blit(txt, (bx - txt.get_width() // 2,
                                   by - txt.get_height() // 2))

        # 爆炸
        for exp in world.explosions:
            ex = exp.x * CELL
            ey = exp.y * CELL
            ci = min(exp.timer, len(C_EXPLODE) - 1)
            c = C_EXPLODE[ci]
            pygame.draw.rect(self.screen, c,
                             (ex + 2, ey + 2, CELL - 4, CELL - 4))
            
        # 角色
        self._draw_fighter(world.player, C_PLAYER, C_PLAYER_D, "P")
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
        self.screen.blit(txt, (fx - txt.get_width() // 2,
                               fy - txt.get_height() // 2))

        # HP条
        hp_w = CELL - 6
        hp_h = 4
        hp_x = fighter.x * CELL + 3
        hp_y = fighter.y * CELL - 6
        pygame.draw.rect(self.screen, (40, 40, 40),
                         (hp_x, hp_y, hp_w, hp_h))
        fill = int(hp_w * fighter.hp / fighter.max_hp)
        hc = C_GOOD if fighter.hp > 1 else C_BAD
        pygame.draw.rect(self.screen, hc, (hp_x, hp_y, fill, hp_h))
        
        # 动作指示器
        act = fighter.last_action
        if act < 4:
            dx, dy = DIR_MAP[act]
            ax = fx + dx * 12
            ay = fy + dy * 12
            pygame.draw.circle(self.screen, c, (ax, ay), 3)

    def draw_panel(self, world, episode, epsilon, loss, mode,
                   speed, ai_wins, player_wins, total_rounds,
                   gate_val, strategy_gen, fps_val):
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

        hearts_p = ("*" * world.player.hp + "." * (MAX_HP - world.player.hp))
        hearts_a = ("*" * world.ai.hp + "." * (MAX_HP - world.ai.hp))
        wr = ai_wins / max(total_rounds, 1) * 100

        infos = [
            ("Mode", mode, C_HIGHLIGHT if mode == "PvAI" else C_GOOD),
            ("Round", f"{episode}", C_TEXT),
            ("Step", f"{world.step_count}/{MAX_ROUND_STEPS}", C_TEXT),
            ("P HP", hearts_p, C_PLAYER),
            ("AI HP", hearts_a, C_AI),
            ("", "", C_TEXT),
            ("P Wins", f"{player_wins}", C_PLAYER),
            ("AI Wins", f"{ai_wins}", C_AI),
            ("WinRate", f"{wr:.1f}%", C_WARN),
            ("Total", f"{total_rounds}", C_DIM),
            ("", "", C_TEXT),
            ("Epsilon", f"{epsilon:.4f}", C_TEXT),
            ("Loss", f"{loss:.5f}", C_TEXT),
            ("Gate", f"{gate_val:.2f}", C_WARN),
            ("Gen", f"{strategy_gen}", C_GOOD),
            ("Speed", f"x{speed}", C_TEXT),
            ("FPS", f"{fps_val:.0f}", C_DIM),
            ("", "", C_TEXT),
            ("Tier", HW.tier, C_HIGHLIGHT),
            ("Hidden", f"{HIDDEN_DIM}", C_DIM),
            ("Batch", f"{BATCH_SIZE}", C_DIM),
        ]
        for label, val, color in infos:
            if label:
                self.screen.blit(
                    self.fonts["sm"].render(f"{label}:", True, C_DIM),
                    (ipx, iy))
                self.screen.blit(
                    self.fonts["sm"].render(str(val), True, color),
                    (ipx + 70, iy))
            iy += 14
            
    def draw_bottom(self, mode):
        by = ARENA_H
        pygame.draw.rect(self.screen, C_BOTTOM,
                         (0, by, WIN_W, BOTTOM_H))
        pygame.draw.line(self.screen, (50, 50, 80),
                         (0, by), (WIN_W, by), 2)

        y1 = by + 8
        y2 = by + 26
        y3 = by + 44
        y4 = by + 60

        self.screen.blit(
            self.fonts["lg"].render(
                f"Grid Duel Arena v{VERSION}  [{HW.tier}]",
                True, C_HIGHLIGHT),
            (10, y1))

        self.screen.blit(
            self.fonts["sm"].render(
                "[Space]Pause [Up/Dn]Speed "
                "[1]PvAI [2]SelfPlay [3]Train",
                True, C_DIM),
            (10, y2))
        self.screen.blit(
            self.fonts["sm"].render(
                "[WASD]Move [J]Bomb [K]Stay "
                "[S]Save [Tab]Help [Esc]Quit",
                True, C_DIM),
            (10, y3))

        dev_str = str(DEVICE)
        if HW.has_cuda:
            dev_str += f" ({HW.gpu_name[:25]}, {HW.vram_mb}MB)"
        self.screen.blit(
            self.fonts["sm"].render(
                f"Mode:{mode} | {dev_str} | ReflexNet",
                True, C_DIM),
            (10, y4))

    def draw_round_result(self, winner, p_wins, ai_wins, total):
        ov = pygame.Surface((ARENA_W, ARENA_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 160))
        self.screen.blit(ov, (0, 0))

        cx = ARENA_W // 2
        cy = ARENA_H // 2

        if winner == "player":
            txt = "PLAYER WINS!"
            color = C_PLAYER
        elif winner == "ai":
            txt = "AI WINS!"
            color = C_AI
        else:
            txt = "DRAW!"
            color = C_HIGHLIGHT

        title = self.fonts["xl"].render(txt, True, color)
        self.screen.blit(title,
                         (cx - title.get_width() // 2, cy - 50))

        score = self.fonts["md"].render(
            f"Player {p_wins}:  {ai_wins} AI(of {total})",
            True, C_TEXT)
        self.screen.blit(score,
                         (cx - score.get_width() // 2, cy + 10))

        hint = self.fonts["sm"].render(
            "[N] Next Round|  [Esc] Quit", True, C_DIM)
        self.screen.blit(hint,
                         (cx - hint.get_width() // 2, cy + 40))

    def draw_help_overlay(self):
        ov = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 200))
        self.screen.blit(ov, (0, 0))

        lines = [
            (f"Grid Duel Arena v{VERSION}", C_HIGHLIGHT, "lg"),
            ("", C_TEXT, "sm"),
            ("--- Controls ---", C_WARN, "md"),
            ("[WASD] Move player", C_TEXT, "md"),
            ("[J] Place bomb  [K] Stay", C_TEXT, "md"),
            ("[Space] Pause  [Up/Down] Speed", C_TEXT, "md"),
            ("[1] PvAI  [2] SelfPlay  [3] FastTrain", C_TEXT, "md"),
            ("[S] Save checkpoint", C_TEXT, "md"),
            ("[Tab] Toggle this help", C_TEXT, "md"),
            ("[Esc] Save & quit", C_TEXT, "md"),
            ("", C_TEXT, "sm"),
            ("--- Architecture ---", C_WARN, "md"),
            ("Dual-channel ReflexNet (Fast+Slow)", C_TEXT, "md"),
            (f"  Fast: 18->48->24->{ACTION_DIM} (danger reflex)",
             C_DIM, "sm"),
            (f"  Slow: {STATE_DIM}->{HIDDEN_DIM}->...->{ACTION_DIM} (strategy)",
             C_DIM, "sm"),
            ("  Learnable gate fuses both channels", C_DIM, "sm"),
            ("  Dueling DQN + N-step + PER", C_DIM, "sm"),
            ("  Immune clone selection (strategy pool)", C_DIM, "sm"),
            ("", C_TEXT, "sm"),
            ("--- Hardware ---", C_WARN, "md"),
            (f"Device: {DEVICE}  Tier: {HW.tier}", C_TEXT, "md"),
            (f"Hidden: {HIDDEN_DIM}  Batch: {BATCH_SIZE}"
             f"Memory: {MEMORY_SIZE}", C_DIM, "sm"),
            (f"AMP: {USE_AMP}  GradAccum: {GRAD_ACCUM}", C_DIM, "sm"),
            ("", C_TEXT, "sm"),
            ("Press [Tab] to close", C_HIGHLIGHT, "md"),
        ]
        y = 25
        for text, color, font_key in lines:
            if text:
                self.screen.blit(
                    self.fonts[font_key].render(text, True, color),
                    (30, y))
            y += 20 if font_key != "lg" else 28


# ╔════════════════════════════════════════════╗
# ║          存档/ 加载 (兼容v1.0)             ║
# ╚════════════════════════════════════════════╝

def migrate_weights(model, old_sd, label=""):
    """安全加载权重，形状不匹配的层跳过"""
    ns = model.state_dict()
    matched = 0
    skipped = 0
    for k in ns:
        if k in old_sd and old_sd[k].shape == ns[k].shape:
            ns[k] = old_sd[k]
            matched += 1
        else:
            skipped += 1
    model.load_state_dict(ns)
    if skipped > 0:
        print(f"  🔄 {label}: {matched} matched, {skipped} adapted")
    return matched, skipped


def save_checkpoint(net, target_net, optimizer, memory, stats,
                    pool, episode, best_reward):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({
        "version": VERSION,
        "episode": episode,
        "best_reward": best_reward,
        "hidden_dim": HIDDEN_DIM,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "hw_tier": HW.tier,
        "net": net.state_dict(),
        "target": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, CKPT_MODEL)

    stats_save = {}
    for k, v in stats.items():
        if isinstance(v, list):
            stats_save[k] = v[-5000:]
        else:
            stats_save[k] = v
    with open(CKPT_STATS, "w") as f:
        json.dump(stats_save, f, indent=2)

    pool.save(CKPT_POOL)
    print(f"  💾 Saved ep={episode} best_r={best_reward:.1f} "
          f"tier={HW.tier}")


def save_best(net, best_reward, episode):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({
        "version": VERSION,
        "episode": episode,
        "best_reward": best_reward,
        "hidden_dim": HIDDEN_DIM,
        "net": net.state_dict(),
    }, CKPT_BEST)
    print(f"🏆 Best model R={best_reward:.1f}")


def load_checkpoint(net, target_net, optimizer, pool):
    default_stats = {
        "ai_wins": 0,
        "player_wins": 0,
        "draws": 0,
        "rewards": [],
        "winrates": [],
        "losses": [],
    }

    if not os.path.exists(CKPT_MODEL):
        print("  🆕 No checkpoint found — starting fresh")
        return 0, -1e9, default_stats

    print(f"  📂 Loading {CKPT_MODEL}...")
    ckpt = torch.load(CKPT_MODEL, map_location=DEVICE, weights_only=False)

    old_ver = ckpt.get("version", "1.0")
    old_hidden = ckpt.get("hidden_dim", 128)

    print(f"  📋 Checkpoint: v{old_ver}, hidden={old_hidden}, "
          f"current: v{VERSION}, hidden={HIDDEN_DIM}")

    if old_hidden == HIDDEN_DIM:
        net.load_state_dict(ckpt["net"])
        tgt = ckpt.get("target", ckpt["net"])
        target_net.load_state_dict(tgt)
        print(f"  ✅ Weights loaded perfectly (same architecture)")
    else:
        print(f"🔄 Architecture changed "
              f"(hidden {old_hidden} → {HIDDEN_DIM}), migrating...")
        migrate_weights(net, ckpt["net"], "net")
        tgt = ckpt.get("target", ckpt["net"])
        migrate_weights(target_net, tgt, "target")

    # 优化器
    if old_hidden == HIDDEN_DIM and old_ver == VERSION:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            print("  ⚠ Optimizer re-initialized")
    else:
        print("  ⚠ Optimizer re-initialized (architecture changed)")

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
# ║            规则AI (内置对手)               ║
# ╚════════════════════════════════════════════╝

def _rule_ai_logic(world, me, enemy):
    """通用规则AI逻辑"""
    in_danger = False
    for b in world.bombs:
        if ((b.x == me.x and abs(b.y - me.y) <= b.power)
                or (b.y == me.y and abs(b.x - me.x) <= b.power)):
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
                if ((b.x == nx and abs(b.y - ny) <= b.power)
                        or (b.y == ny and abs(b.x - nx) <= b.power)):
                    safe = False
                safety += abs(b.x - nx) + abs(b.y - ny)
            safety += 10 if safe else 0
            if safety > best_safety:
                best_safety = safety
                best_dir = act
        return best_dir

    dist = abs(me.x - enemy.x) + abs(me.y - enemy.y)
    if (dist <= 3
            and me.bomb_cooldown <= 0
            and me.active_bombs < me.max_bombs):
        can_escape = False
        for act in range(4):
            dx, dy = DIR_MAP[act]
            nx, ny = me.x + dx, me.y + dy
            if (world._can_move(nx, ny)
                    and not (nx == enemy.x and ny == enemy.y)):
                can_escape = True
                break
        if can_escape:
            return 4

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


def rule_based_ai(world):
    return _rule_ai_logic(world, world.ai, world.player)


def rule_based_player(world):
    act = _rule_ai_logic(world, world.player, world.ai)
    if random.random() < 0.12:
        act = random.randint(0, 5)
    return act


# ╔════════════════════════════════════════════╗
# ║          AMP Scaler (自适应)               ║
# ╚════════════════════════════════════════════╝

class AMPContext:
    """根据硬件自动选择是否用混合精度"""

    def __init__(self):
        self.use_amp = USE_AMP and DEVICE.type == "cuda"
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def autocast(self):
        if self.use_amp:
            return torch.cuda.amp.autocast()
        return NullContext()

    def scale_and_step(self, loss, optimizer, params, max_norm=10.0):
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


class NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ╔════════════════════════════════════════════╗
# ║              主程序入口                    ║
# ╚════════════════════════════════════════════╝

def main():
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

    # ——— 模型 ———
    net = ReflexNet(hidden=HIDDEN_DIM).to(DEVICE)
    target_net = ReflexNet(hidden=HIDDEN_DIM).to(DEVICE)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    memory = LightPER(MEMORY_SIZE)
    nstep = NStepBuffer(N_STEP, GAMMA)
    pool = StrategyPool(max_size=8)

    opponent_net = ReflexNet(hidden=HIDDEN_DIM).to(DEVICE)
    opponent_net.eval()

    amp_ctx = AMPContext()

    # ——— 加载 ———
    start_ep, best_reward, stats = load_checkpoint(
        net, target_net, optimizer, pool)

    renderer = Renderer(screen, clock, fonts)
    for v in stats.get("winrates", [])[-150:]:
        renderer.chart_winrate.add(v)
    for v in stats.get("rewards", [])[-150:]:
        renderer.chart_reward.add(v)
    for v in stats.get("losses", [])[-150:]:
        renderer.chart_eps.add(v)

    # ——— 状态 ———
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
    grad_accum_counter = 0

    print(f"\n  🏟️  Grid Duel Arena v{VERSION}")
    print(f"     Device: {DEVICE} | Tier: {HW.tier}")
    print(f"     [1] PvAI  [2] SelfPlay  [3] FastTrain\n")

    episode = start_ep

    # ═══════════ 主循环 ═══════════
    running = True
    while running:
        world = DuelArena(seed=random.randint(0, 2**31))
        obs = world.reset()
        total_reward = 0.0
        nstep.reset()
        show_result = False
        result_timer = 0

        epsilon = max(EPS_END, EPS_START - global_step / EPS_DECAY)

        # 对手选择
        use_pool_opp = (mode in ("SelfPlay", "Train")
                        and pool.pool
                        and random.random() < 0.3)
        if use_pool_opp:
            opp = pool.sample_opponent()
            try:
                migrate_weights(opponent_net, opp[1], "opp")
            except Exception:
                use_pool_opp = False
            opp_name = opp[0] if use_pool_opp else "Rule"
        else:
            opp_name = "Rule"

        round_running = True
        while round_running and running:
            # ——— 事件 ———
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
                        speed = 15
                    elif ev.key == pygame.K_TAB:
                        show_help = not show_help
                    elif ev.key == pygame.K_n and show_result:
                        round_running = False
                        continue
                    elif ev.key == pygame.K_s and not show_result:
                        stats["player_wins"] = player_wins
                        stats["ai_wins"] = ai_wins
                        stats["draws"] = draws
                        save_checkpoint(net, target_net, optimizer,
                                        memory, stats, pool,
                                        episode, best_reward)
                    # 玩家操作 (PvAI模式)
                    if mode == "PvAI" and not show_result:
                        if ev.key == pygame.K_w:
                            player_action_queue = 0
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

            # 持续按键检测 (改善操控手感)
            if mode == "PvAI" and not show_result and not paused:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_w]:
                    player_action_queue = 0
                elif keys[pygame.K_s]:
                    player_action_queue = 1
                elif keys[pygame.K_a]:
                    player_action_queue = 2
                elif keys[pygame.K_d]:
                    player_action_queue = 3

            fps_val = clock.get_fps()

            # ——— 暂停 ———
            if paused and not show_result:
                renderer.draw_arena(world)
                renderer.draw_panel(world, episode, epsilon, last_loss, mode, speed,
                                    ai_wins, player_wins,
                                    total_rounds, gate_val,
                                    pool.generation, fps_val)
                renderer.draw_bottom(mode)
                ptxt = fonts["lg"].render("PAUSED", True, C_WARN)
                screen.blit(ptxt, (ARENA_W // 2 - ptxt.get_width() // 2, ARENA_H // 2 - ptxt.get_height() // 2))
                if show_help:
                    renderer.draw_help_overlay()
                pygame.display.flip()
                clock.tick(15)
                continue

            # ——— 结算画面 ———
            if show_result:
                renderer.draw_arena(world)
                renderer.draw_panel(world, episode, epsilon,
                                    last_loss, mode, speed,
                                    ai_wins, player_wins,
                                    total_rounds, gate_val,
                                    pool.generation, fps_val)
                renderer.draw_bottom(mode)
                renderer.draw_round_result(world.winner, player_wins, ai_wins,
                                           total_rounds)
                if show_help:
                    renderer.draw_help_overlay()
                pygame.display.flip()
                clock.tick(15)
                if mode in ("SelfPlay", "Train"):
                    result_timer += 1
                    limit = 2 if mode == "Train" else 12
                    if result_timer > limit:
                        round_running = False
                continue

            # ——— 决定玩家动作 ———
            if mode == "PvAI":
                p_act = player_action_queue
                player_action_queue = 5
            elif use_pool_opp:
                st = world.get_state(for_ai=False)
                st_t = torch.tensor(st, dtype=torch.float32,
                                    device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    q = opponent_net(st_t)
                p_act = q.argmax(dim=-1).item()
            else:
                p_act = rule_based_player(world)

            # ——— AI决策 ———
            obs_t = torch.tensor(obs, dtype=torch.float32,
                                 device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q_values = net(obs_t)
                gate_val = net.get_gate_value(obs_t)

            if random.random() < epsilon:
                ai_act = random.randint(0, ACTION_DIM - 1)
            else:
                ai_act = q_values.argmax(dim=-1).item()

            # ——— 执行 ———
            prev_ai_hp = world.ai.hp
            prev_player_hp = world.player.hp
            obs2, reward, done = world.step(p_act, ai_act)
            total_reward += reward
            global_step += 1

            # 粒子效果
            for exp in world.new_explosions:
                renderer.add_explosion_particles(exp.x, exp.y)
            if world.ai.hp < prev_ai_hp:
                renderer.add_hit_particles(world.ai.x, world.ai.y, C_AI)
            if world.player.hp < prev_player_hp:
                renderer.add_hit_particles(world.player.x,
                                           world.player.y, C_PLAYER)

            # 存储经验
            nstep.push((obs, ai_act, reward, obs2, float(done)))
            nt = nstep.get()
            if nt:
                memory.push(nt)
            obs = obs2

            # ——— 训练 ———
            train_iters = TRAIN_PER_FRAME if not done else 1
            for _ in range(train_iters):
                if len(memory) < BATCH_SIZE:
                    break

                per_beta = min(1.0, 0.4 + global_step * 0.0001)
                batch, idx, isw = memory.sample(BATCH_SIZE, per_beta)

                bs = torch.tensor(
                    np.array([t[0] for t in batch]),
                    dtype=torch.float32, device=DEVICE)
                ba = torch.tensor(
                    [t[1] for t in batch],
                    dtype=torch.long, device=DEVICE).unsqueeze(-1)
                br = torch.tensor(
                    [t[2] for t in batch],
                    dtype=torch.float32, device=DEVICE).unsqueeze(-1)
                bs2 = torch.tensor(
                    np.array([t[3] for t in batch]),
                    dtype=torch.float32, device=DEVICE)
                bd = torch.tensor(
                    [t[4] for t in batch],
                    dtype=torch.float32, device=DEVICE).unsqueeze(-1)

                with amp_ctx.autocast():
                    with torch.no_grad():
                        best_a = net(bs2).argmax(dim=-1, keepdim=True)
                        q_next = target_net(bs2).gather(1, best_a)
                        target = (br + GAMMA ** N_STEP
                                  * q_next * (1 - bd))

                    q_current = net(bs).gather(1, ba)
                    td_error = ((target - q_current)
                                .detach().squeeze().cpu().numpy())
                    loss = (isw.unsqueeze(-1).to(DEVICE) * (q_current - target) ** 2).mean()

                    if GRAD_ACCUM > 1:
                        loss = loss / GRAD_ACCUM

                optimizer.zero_grad()
                amp_ctx.scale_and_step(loss, optimizer, net.parameters(), 10.0)
                memory.update_priorities(idx, td_error)
                last_loss = loss.item() * (GRAD_ACCUM if GRAD_ACCUM > 1 else 1)
                
                # 软更新
                for tp, sp in zip(target_net.parameters(), net.parameters()):
                    tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)

            # ——— 渲染 ———
            do_render = True
            if mode == "Train" and global_step % 4 != 0:
                do_render = False

            if do_render:
                renderer.draw_arena(world)
                renderer.draw_panel(world, episode, epsilon,
                                    last_loss, mode, speed,
                                    ai_wins, player_wins,
                                    total_rounds, gate_val,
                                    pool.generation, fps_val)
                renderer.draw_bottom(mode)
                if show_help:
                    renderer.draw_help_overlay()
                pygame.display.flip()

            fps = BASE_FPS * speed
            if mode == "Train":
                fps = max(fps, MAX_FPS_TRAIN)
            clock.tick(fps)

            # ——— 回合结束 ———
            if done:
                for t in nstep.flush():
                    memory.push(t)
                show_result = True
                result_timer = 0
                total_rounds += 1

                if world.winner == "ai":
                    ai_wins += 1
                elif world.winner == "player":
                    player_wins += 1
                else:
                    draws += 1

                wr = ai_wins / max(total_rounds, 1) * 100
                stats.setdefault("rewards", []).append(total_reward)
                stats.setdefault("winrates", []).append(wr)
                stats.setdefault("losses", []).append(last_loss)

                renderer.chart_winrate.add(wr)
                renderer.chart_reward.add(total_reward)
                renderer.chart_eps.add(epsilon)

                winner_str = {
                    "ai": "AI WIN",
                    "player": "P WIN",
                    "draw": "DRAW",
                }.get(world.winner, "?")

                print(f"EP {episode:5d}|"
                      f"{winner_str:>6s}|"
                      f"R:{total_reward:7.1f}|"
                      f"AI:{ai_wins} P:{player_wins}|"
                      f"WR:{wr:5.1f}%|"
                      f"e:{epsilon:.3f}|"
                      f"G:{gate_val:.2f}|"
                      f"Opp:{opp_name}|"
                      f"Mem:{len(memory)}")

                # 策略池
                if episode > 0 and episode % 20 == 0:
                    rr = stats["rewards"]
                    recent = rr[-20:] if len(rr) >= 20 else rr
                    fitness = sum(recent) / max(len(recent), 1)
                    pool.add(f"gen{pool.generation}_ep{episode}",
                             net.state_dict(), fitness)

                if total_reward > best_reward:
                    best_reward = total_reward
                    save_best(net, best_reward, episode)

                if (episode + 1) % 50 == 0:
                    stats["player_wins"] = player_wins
                    stats["ai_wins"] = ai_wins
                    stats["draws"] = draws
                    save_checkpoint(net, target_net, optimizer,
                                    memory, stats, pool,
                                    episode + 1, best_reward)

                episode += 1

    # ——— 退出保存 ———
    stats["player_wins"] = player_wins
    stats["ai_wins"] = ai_wins
    stats["draws"] = draws
    save_checkpoint(net, target_net, optimizer, memory, stats,
                    pool, episode, best_reward)
    pygame.quit()
    print("\n  👋 Game saved. Goodbye!")

if __name__ == "__main__":
    main()