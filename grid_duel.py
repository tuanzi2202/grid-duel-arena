#!/usr/bin/env python3
"""
Grid Duel Arena v4.1 — Rainbow DQN + Full PvAI Experience
==========================================================
★ Rainbow DQN: C51 + NoisyNet + Dueling + Double + SumTree-PER + N-step
★ ResidualAttentionNet: 512隐 + 多头注意力 + 残差块
★ 完整PvAI:玩家WASD+JK对战训练好的AI，支持难度切换
★ BatchedEnvWorker: 每worker跑多个env，批量推理
★ BFS势函数奖励塑形，理论安全
★ torch.compile + CUDA Stream + AMP
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

VERSION = "4.1"

# ═══════════════════════════════════════
# 竞技场常量
# ═══════════════════════════════════════
ARENA_COLS = 13
ARENA_ROWS = 11
CELL = 40
ARENA_W = ARENA_COLS * CELL
ARENA_H = ARENA_ROWS * CELL
PANEL_W = 260
BOTTOM_H = 80
WIN_W = ARENA_W + PANEL_W
WIN_H = ARENA_H + BOTTOM_H

STATE_DIM = 58
ACTION_DIM = 6

# Rainbow超参数
GAMMA = 0.99
LR = 3e-4
LR_MIN = 5e-6
TAU = 0.005
N_STEP = 5
V_MIN = -30.0
V_MAX = 30.0
N_ATOMS = 51
EVAL_INTERVAL = 25

MAX_HP = 3
BOMB_TIMER = 8
BOMB_RANGE = 2
BOMB_COOLDOWN = 12
POWERUP_INTERVAL = 60
MAX_ROUND_STEPS = 500
BASE_FPS = 8

CKPT_DIR = "grid_duel_ckpt_v4"
CKPT_MODEL = os.path.join(CKPT_DIR, "model.pth")
CKPT_BEST = os.path.join(CKPT_DIR, "best.pth")
CKPT_POOL = os.path.join(CKPT_DIR, "strategy_pool.pkl")
CKPT_STATS = os.path.join(CKPT_DIR, "stats.json")
WEIGHT_FILE = os.path.join(CKPT_DIR, ".worker_weights.pt")

DIR_MAP = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0), 5: (0, 0)}
DIR_NAMES = ["Up", "Dn", "Lt", "Rt", "Bomb", "Stay"]

EMPTY, WALL, BRICK = 0, 1, 2

# 颜色
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

# AI 难度
DIFFICULTY_LEVELS = {
    "Easy": {"noise_scale": 2.0, "random_prob": 0.15, "label": "🟢 Easy"},
    "Normal": {"noise_scale": 1.0, "random_prob": 0.05, "label": "🟡 Normal"},
    "Hard": {"noise_scale": 0.3, "random_prob": 0.01, "label": "🔴 Hard"},
    "Expert": {"noise_scale": 0.0, "random_prob": 0.00, "label": "💀 Expert"},
}
DIFFICULTY_ORDER = ["Easy", "Normal", "Hard", "Expert"]


# ═══════════════════════════════════════
#  竞技场 & 战斗逻辑
# ═══════════════════════════════════════

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
        self.prev_hp = MAX_HP

    def reset(self, x, y):
        self.x = x
        self.y = y
        self.hp = self.max_hp
        self.prev_hp = self.max_hp
        self.bomb_cooldown = 0
        self.active_bombs = 0
        self.invincible = 0
        self.last_action = 5


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
        self._prev_potential = None
        self._ai_prev_safe_score = 0.0
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
        self._prev_potential = self._compute_potential()
        self._ai_prev_safe_score = self._bfs_safety(self.ai.x, self.ai.y)
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
            if fighter.bomb_cooldown <= 0 and fighter.active_bombs < fighter.max_bombs:
                has_bomb = any(b.x == fighter.x and b.y == fighter.y for b in self.bombs)
                if not has_bomb:
                    self.bombs.append(Bomb(fighter.x, fighter.y, fighter, fighter.bomb_power))
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
                    fighter.x = nx
                    fighter.y = ny
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
                        self.powerups.append(PowerUp(ex, ey, self.rng.randint(0, 2)))
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
                        other = self.ai if fighter is self.player else self.player
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

    def _spawn_random_powerup(self):
        occupied = {(self.player.x, self.player.y), (self.ai.x, self.ai.y)}
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
            self.powerups.append(PowerUp(x, y, self.rng.randint(0, 2)))

    def _bfs_safety(self, sx, sy, max_depth=4):
        danger_set = set()
        for b in self.bombs:
            danger_set.add((b.x, b.y))
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                for r in range(1, b.power + 1):
                    ex, ey = b.x + dx * r, b.y + dy * r
                    if ex < 0 or ex >= ARENA_COLS or ey < 0 or ey >= ARENA_ROWS:
                        break
                    if self.grid[ey][ex] == WALL:
                        break
                    danger_set.add((ex, ey))
                    if self.grid[ey][ex] == BRICK:
                        break
        for exp in self.explosions:
            danger_set.add((exp.x, exp.y))
        visited = {(sx, sy)}
        queue = deque([(sx, sy, 0)])
        safe_count = 1 if (sx, sy) not in danger_set else 0
        while queue:
            x, y, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited:
                    continue
                if not self._can_move(nx, ny):
                    continue
                visited.add((nx, ny))
                if (nx, ny) not in danger_set:
                    safe_count += 1
                queue.append((nx, ny, depth + 1))
        return safe_count

    def _compute_potential(self):
        me = self.ai
        enemy = self.player
        if me.hp <= 0:
            return -50.0
        if enemy.hp <= 0:
            return 50.0
        phi = 0.0
        phi += (me.hp - enemy.hp) * 8.0
        dist = abs(me.x - enemy.x) + abs(me.y - enemy.y)
        if 2 <= dist <= 4:
            phi += 3.0
        elif dist <= 1:
            phi -= 2.0
        elif dist >= 8:
            phi -= dist * 0.3
        safe = self._bfs_safety(me.x, me.y)
        phi += safe * 0.5
        phi += (me.bomb_power - enemy.bomb_power) * 1.5
        phi += (me.max_bombs - enemy.max_bombs) * 2.0
        cx = abs(me.x - ARENA_COLS // 2)
        cy = abs(me.y - ARENA_ROWS // 2)
        phi += max(0, 5 - cx - cy) * 0.3
        return phi

    def _compute_advanced_reward(self):
        reward = 0.0
        new_potential = self._compute_potential()
        if self._prev_potential is not None:
            reward += GAMMA * new_potential - self._prev_potential
        self._prev_potential = new_potential
        me = self.ai
        enemy = self.player
        dist = abs(me.x - enemy.x) + abs(me.y - enemy.y)
        if enemy.hp < enemy.prev_hp:
            reward += 5.0
        if me.hp < me.prev_hp:
            reward -= 4.0
        new_safe = self._bfs_safety(me.x, me.y)
        safe_delta = new_safe - self._ai_prev_safe_score
        reward += safe_delta * 0.15
        self._ai_prev_safe_score = new_safe
        for b in self.bombs:
            in_cross = ((b.x == me.x and abs(b.y - me.y) <= b.power) or
                        (b.y == me.y and abs(b.x - me.x) <= b.power))
            if in_cross:
                urgency = 1.0 - b.timer / BOMB_TIMER
                reward -= (1.0 * urgency + 0.3)
        for exp in self.explosions:
            if me.x == exp.x and me.y == exp.y:
                reward -= 1.0
        if me.last_action == 4 and self.last_ai_move_valid:
            enemy_in_cross = ((me.x == enemy.x and abs(me.y - enemy.y) <= me.bomb_power) or
                              (me.y == enemy.y and abs(me.x - enemy.x) <= me.bomb_power))
            if enemy_in_cross and dist <= me.bomb_power + 1:
                reward += 1.5
            elif dist <= 3:
                reward += 0.3
            else:
                reward -= 0.5
            post_safe = self._bfs_safety(me.x, me.y, max_depth=3)
            if post_safe < 2:
                reward -= 1.0
        if not self.last_ai_move_valid:
            reward -= 0.1
        if self.bricks_broken_by_ai > 0:
            reward += 0.15 * self.bricks_broken_by_ai
            self.bricks_broken_by_ai = 0
        escapes = sum(1 for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]
                      if self._can_move(me.x + dx, me.y + dy))
        if escapes == 0:
            reward -= 0.5
        reward += 0.002
        return np.clip(reward, -10.0, 10.0)

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
                                if other_bomb.x == cx and other_bomb.y == cy:
                                    other_bomb.timer = 0
                    bomb.owner.active_bombs = max(0, bomb.owner.active_bombs - 1)
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
            ai_reward = -5.0
        elif self.player.hp <= 0:
            self.winner = "ai"
            self.round_over = True
            done = True
            ai_reward = 25.0
        elif self.ai.hp <= 0:
            self.winner = "player"
            self.round_over = True
            done = True
            ai_reward = -25.0
        elif self.step_count >= MAX_ROUND_STEPS:
            if self.ai.hp > self.player.hp:
                self.winner = "ai"
                ai_reward = 12.0
            elif self.player.hp > self.ai.hp:
                self.winner = "player"
                ai_reward = -12.0
            else:
                self.winner = "draw"
                ai_reward = -3.0
            self.round_over = True
            done = True
        if not done:
            ai_reward = self._compute_advanced_reward()
        return self.get_state(for_ai=True), ai_reward, done

    def get_state(self, for_ai=True):
        me = self.ai if for_ai else self.player
        enemy = self.player if for_ai else self.ai
        f = []
        f.append(me.x / ARENA_COLS)
        f.append(me.y / ARENA_ROWS)
        f.append(me.hp / me.max_hp)
        f.append(me.bomb_cooldown / BOMB_COOLDOWN)
        f.append(me.active_bombs / max(me.max_bombs, 1))
        f.append(me.invincible / 6.0)
        f.append((enemy.x - me.x) / ARENA_COLS)
        f.append((enemy.y - me.y) / ARENA_ROWS)
        f.append(enemy.hp / enemy.max_hp)
        dist = abs(enemy.x - me.x) + abs(enemy.y - me.y)
        f.append(dist / (ARENA_COLS + ARENA_ROWS))
        angle = math.atan2(enemy.y - me.y, enemy.x - me.x)
        f.append(angle / math.pi)
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in dirs:
            wall_dist = 0
            for r in range(1, max(ARENA_COLS, ARENA_ROWS)):
                cx_, cy_ = me.x + dx * r, me.y + dy * r
                if cx_ < 0 or cx_ >= ARENA_COLS or cy_ < 0 or cy_ >= ARENA_ROWS:
                    break
                if self.grid[cy_][cx_] in (WALL, BRICK):
                    break
                wall_dist = r
            f.append(wall_dist / max(ARENA_COLS, ARENA_ROWS))
            danger = 0.0
            for b in self.bombs:
                if dx != 0 and b.y == me.y and 0 < (b.x - me.x) * dx <= b.power + 1:
                    danger = max(danger, 1.0 - b.timer / BOMB_TIMER)
                if dy != 0 and b.x == me.x and 0 < (b.y - me.y) * dy <= b.power + 1:
                    danger = max(danger, 1.0 - b.timer / BOMB_TIMER)
            f.append(danger)
            has_exp = 0.0
            for exp in self.explosions:
                if dx != 0 and exp.y == me.y and 0 < (exp.x - me.x) * dx <= 2:
                    has_exp = 1.0
                if dy != 0 and exp.x == me.x and 0 < (exp.y - me.y) * dy <= 2:
                    has_exp = 1.0
            f.append(has_exp)
        my_bc = sum(1 for b in self.bombs if b.owner is me)
        en_bc = sum(1 for b in self.bombs if b.owner is enemy)
        f.append(my_bc / 3.0)
        f.append(en_bc / 3.0)
        min_bd = 1.0
        min_bt = 1.0
        for b in self.bombs:
            bd = (abs(b.x - me.x) + abs(b.y - me.y)) / (ARENA_COLS + ARENA_ROWS)
            if bd < min_bd:
                min_bd = bd
                min_bt = b.timer / BOMB_TIMER
        f.append(min_bd)
        f.append(min_bt)
        in_danger = 0.0
        for b in self.bombs:
            if ((b.x == me.x and abs(b.y - me.y) <= b.power) or
                    (b.y == me.y and abs(b.x - me.x) <= b.power)):
                in_danger = max(in_danger, 1.0 - b.timer / BOMB_TIMER)
        f.append(in_danger)
        stuck = 1.0
        escape_dirs = 0
        for ddx, ddy in dirs:
            if self._can_move(me.x + ddx, me.y + ddy):
                stuck = 0.0
                escape_dirs += 1
        f.append(stuck)
        f.append(escape_dirs / 4.0)
        min_pd = 1.0
        min_pdx = 0.0
        min_pdy = 0.0
        for pu in self.powerups:
            pd = (abs(pu.x - me.x) + abs(pu.y - me.y)) / (ARENA_COLS + ARENA_ROWS)
            if pd < min_pd:
                min_pd = pd
                min_pdx = (pu.x - me.x) / ARENA_COLS
                min_pdy = (pu.y - me.y) / ARENA_ROWS
        f.append(min_pd)
        f.append(min_pdx)
        f.append(min_pdy)
        f.append(self.step_count / MAX_ROUND_STEPS)
        f.append(me.bomb_power / 5.0)
        f.append(me.max_bombs / 3.0)
        f.append(len(self.powerups) / 4.0)
        f.append(enemy.bomb_power / 5.0)
        f.append(enemy.bomb_cooldown / BOMB_COOLDOWN)
        f.append(enemy.active_bombs / max(enemy.max_bombs, 1))
        f.append(enemy.invincible / 6.0)
        en_escapes = sum(1 for dx, dy in dirs if self._can_move(enemy.x + dx, enemy.y + dy))
        f.append(en_escapes / 4.0)
        f.append(1.0 if me.x == enemy.x else 0.0)
        f.append(1.0 if me.y == enemy.y else 0.0)
        line_clear = 1.0
        if me.x == enemy.x:
            for yy in range(min(me.y, enemy.y) + 1, max(me.y, enemy.y)):
                if self.grid[yy][me.x] in (WALL, BRICK):
                    line_clear = 0.0
                    break
        elif me.y == enemy.y:
            for xx in range(min(me.x, enemy.x) + 1, max(me.x, enemy.x)):
                if self.grid[me.y][xx] in (WALL, BRICK):
                    line_clear = 0.0
                    break
        else:
            line_clear = 0.0
        f.append(line_clear)
        safe = self._bfs_safety(me.x, me.y, max_depth=3)
        f.append(min(safe / 12.0, 1.0))
        en_safe = self._bfs_safety(enemy.x, enemy.y, max_depth=3)
        f.append(min(en_safe / 12.0, 1.0))
        can_hit = 0.0
        if me.bomb_cooldown <= 0 and me.active_bombs < me.max_bombs:
            if ((me.x == enemy.x and abs(me.y - enemy.y) <= me.bomb_power) or
                    (me.y == enemy.y and abs(me.x - enemy.x) <= me.bomb_power)):
                can_hit = 1.0
        f.append(can_hit)
        for dx, dy in dirs:
            f.append(1.0 if self._can_move(me.x + dx, me.y + dy) else 0.0)
        f.append(1.0 if (me.bomb_cooldown <= 0 and me.active_bombs < me.max_bombs) else 0.0)
        f.append((me.hp - enemy.hp) / MAX_HP)
        f.append(len(self.bombs) / 6.0)
        f.append(me.last_action / 5.0)
        while len(f) < STATE_DIM:
            f.append(0.0)
        return np.array(f[:STATE_DIM], dtype=np.float32)

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
                        dmap[ey][ex] = max(dmap[ey][ex], urgency * 0.7)
                    else:
                        break
        for exp in self.explosions:
            if 0 <= exp.y < ARENA_ROWS and 0 <= exp.x < ARENA_COLS:
                dmap[exp.y][exp.x] = 1.0
        return dmap


# ═══════════════════════════════════════
#  NoisyLinear
# ═══════════════════════════════════════

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ═══════════════════════════════════════
#  RainbowNet — C51 分布式网络
# ═══════════════════════════════════════

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x) * 0.1


class MultiHeadGate(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, D = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, 3, self.heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q * k).sum(-1, keepdim=True) / math.sqrt(self.head_dim)
        attn = torch.sigmoid(attn)
        out = (v * attn).reshape(B, D)
        return residual + self.proj(out) * 0.1


class RainbowNet(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden=512, n_atoms=N_ATOMS):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.n_atoms = n_atoms

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            Swish(),
            nn.LayerNorm(hidden),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden),
            MultiHeadGate(hidden, heads=4),
            ResidualBlock(hidden),
            MultiHeadGate(hidden, heads=4),
            ResidualBlock(hidden),
        )
        tac_dim = 20
        self.tactical_encoder = nn.Sequential(
            nn.Linear(tac_dim, 64),
            Swish(),
            nn.Linear(64, hidden // 4),
            Swish(),
        )
        fused_dim = hidden + hidden // 4
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            Swish(),
            nn.LayerNorm(hidden),
        )
        self.value_hidden = NoisyLinear(hidden, hidden // 2)
        self.value_out = NoisyLinear(hidden // 2, n_atoms)
        self.adv_hidden = NoisyLinear(hidden, hidden // 2)
        self.adv_out = NoisyLinear(hidden // 2, action_dim * n_atoms)

    def _extract_tactical(self, x):
        B = x.shape[0]
        pieces = []
        if x.shape[-1] > 54:
            pieces.append(x[:, 29:32])
            pieces.append(x[:, 11:23])
            pieces.append(x[:, 9:10])
            pieces.append(x[:, 49:50])
            pieces.append(x[:, 54:55])
            pieces.append(x[:, 55:56])
        else:
            pieces.append(x[:, :20])
        tac = torch.cat(pieces, dim=-1)
        if tac.shape[-1] < 20:
            pad = torch.zeros(B, 20 - tac.shape[-1], device=x.device)
            tac = torch.cat([tac, pad], dim=-1)
        return tac[:, :20]

    def forward(self, x):
        B = x.shape[0]
        h = self.encoder(x)
        h = self.res_blocks(h)
        tac = self._extract_tactical(x)
        tac_h = self.tactical_encoder(tac)
        h = self.fusion(torch.cat([h, tac_h], dim=-1))
        v = F.relu(self.value_hidden(h))
        v = self.value_out(v).view(B, 1, self.n_atoms)
        a = F.relu(self.adv_hidden(h))
        a = self.adv_out(a).view(B, self.action_dim, self.n_atoms)
        q_logits = v + a - a.mean(dim=1, keepdim=True)
        return F.log_softmax(q_logits, dim=-1)

    def reset_noise(self):
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.adv_hidden.reset_noise()
        self.adv_out.reset_noise()

    def get_q_values(self, x, support):
        log_probs = self(x)
        probs = log_probs.exp()
        return (probs * support.unsqueeze(0).unsqueeze(0)).sum(-1)

    def get_q_values_np(self, state_np, device, support):
        with torch.no_grad():
            st = torch.from_numpy(state_np).unsqueeze(0).to(device)
            return self.get_q_values(st, support).squeeze(0).cpu().numpy()


# ═══════════════════════════════════════
#  SumTree PER
# ═══════════════════════════════════════

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority):
        idx = self.data_pointer + self.capacity - 1
        self.update(idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx


class SumTreePER:
    def __init__(self, capacity, state_dim=STATE_DIM, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self._max_priority = 1.0

    def push(self, transition):
        s, a, r, ns, d = transition
        self.states[self.pos] = s
        self.actions[self.pos] = a
        self.rewards[self.pos] = r
        self.next_states[self.pos] = ns
        self.dones[self.pos] = d
        self.tree.add(self._max_priority ** self.alpha)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, states, actions, rewards, next_states, dones):
        n = len(states)
        for i in range(n):
            self.states[self.pos] = states[i]
            self.actions[self.pos] = actions[i]
            self.rewards[self.pos] = rewards[i]
            self.next_states[self.pos] = next_states[i]
            self.dones[self.pos] = dones[i]
            self.tree.add(self._max_priority ** self.alpha)
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        batch_size = min(batch_size, self.size)
        indices = np.zeros(batch_size, dtype=np.int64)
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            tidx, p, didx = self.tree.get(s)
            tree_indices[i] = tidx
            priorities[i] = max(p, 1e-8)
            indices[i] = didx % self.capacity
        p_min = priorities.min() / max(self.tree.total, 1e-8)
        max_weight = max((p_min * self.size) ** (-beta), 1e-8)
        weights = (priorities / max(self.tree.total, 1e-8) * self.size) ** (-beta)
        weights /= max_weight
        return (
            self.states[indices].copy(),
            self.actions[indices].copy(),
            self.rewards[indices].copy(),
            self.next_states[indices].copy(),
            self.dones[indices].copy(),
        ), tree_indices, torch.from_numpy(weights.astype(np.float32))

    def update_priorities(self, tree_indices, td_errors):
        pris = np.abs(td_errors) + 1e-6
        for i, tidx in enumerate(tree_indices):
            self.tree.update(int(tidx), pris[i] ** self.alpha)
            self._max_priority = max(self._max_priority, pris[i])

    def __len__(self):
        return self.size


# ═══════════════════════════════════════
#  工具类
# ═══════════════════════════════════════

class NStepBuffer:
    def __init__(self, n=N_STEP, gamma=GAMMA):
        self.n = n
        self.gamma = gamma
        self.buffer = deque(maxlen=n)

    def push(self, t):
        self.buffer.append(t)

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


class StrategyPool:
    def __init__(self, max_size=12):
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
        if random.random() < 0.6:
            return self.best()
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
            except Exception:
                pass


def _rule_ai_logic(world, me, enemy):
    in_danger = False
    for b in world.bombs:
        if ((b.x == me.x and abs(b.y - me.y) <= b.power) or
                (b.y == me.y and abs(b.x - me.x) <= b.power)):
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
            safe = True
            safety = 0
            for b in world.bombs:
                if ((b.x == nx and abs(b.y - ny) <= b.power) or
                        (b.y == ny and abs(b.x - nx) <= b.power)):
                    safe = False
                safety += abs(b.x - nx) + abs(b.y - ny)
            safety += 10 if safe else 0
            if safety > best_safety:
                best_safety = safety
                best_dir = act
        return best_dir
    dist = abs(me.x - enemy.x) + abs(me.y - enemy.y)
    if dist <= 3 and me.bomb_cooldown <= 0 and me.active_bombs < me.max_bombs:
        for act in range(4):
            dx, dy = DIR_MAP[act]
            nx, ny = me.x + dx, me.y + dy
            if world._can_move(nx, ny) and not (nx == enemy.x and ny == enemy.y):
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
            best_dist = nd
            best_dir = act
    return best_dir


def rule_based_player(world):
    act = _rule_ai_logic(world, world.player, world.ai)
    if random.random() < 0.12:
        act = random.randint(0, 5)
    return act


# ═══════════════════════════════════════
#  AMP /训练
# ═══════════════════════════════════════

class NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class AMPContext:
    def __init__(self, use_amp, device):
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def autocast(self):
        if self.use_amp:
            return torch.amp.autocast("cuda")
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


def compute_c51_loss(net, target_net, batch, support, delta_z, device, isw, gamma_n):
    s_np, a_np, r_np, s2_np, d_np = batch
    bs = torch.from_numpy(s_np).to(device, non_blocking=True)
    ba = torch.from_numpy(a_np).to(device, non_blocking=True).long()
    br = torch.from_numpy(r_np).to(device, non_blocking=True)
    bs2 = torch.from_numpy(s2_np).to(device, non_blocking=True)
    bd = torch.from_numpy(d_np).to(device, non_blocking=True)
    B = bs.shape[0]
    log_probs = net(bs)
    log_probs_a = log_probs[torch.arange(B, device=device), ba]
    with torch.no_grad():
        next_log_online = net(bs2)
        next_probs_online = next_log_online.exp()
        next_q_online = (next_probs_online * support.unsqueeze(0).unsqueeze(0)).sum(-1)
        next_actions = next_q_online.argmax(dim=1)
        next_log_target = target_net(bs2)
        next_probs_target = next_log_target.exp()
        next_probs_a = next_probs_target[torch.arange(B, device=device), next_actions]
        Tz = br.unsqueeze(1) + gamma_n * (1 - bd.unsqueeze(1)) * support.unsqueeze(0)
        Tz = Tz.clamp(V_MIN, V_MAX)
        b_idx = (Tz - V_MIN) / delta_z
        l = b_idx.floor().long().clamp(0, N_ATOMS - 1)
        u = b_idx.ceil().long().clamp(0, N_ATOMS - 1)
        m = torch.zeros(B, N_ATOMS, device=device)
        m.scatter_add_(1, l, (u.float() - b_idx) * next_probs_a)
        m.scatter_add_(1, u, (b_idx - l.float()) * next_probs_a)
    loss_per_sample = -(m * log_probs_a).sum(-1)
    weighted_loss = (isw.to(device) * loss_per_sample).mean()
    td_error = loss_per_sample.detach().cpu().numpy()
    return weighted_loss, td_error


# ═══════════════════════════════════════
#  Worker & Collector
# ═══════════════════════════════════════

def _batched_env_worker(worker_id, transition_queue, result_queue, weight_version, stop_flag, config):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    local_device = torch.device("cpu")
    local_net = RainbowNet(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        hidden=config["hidden_dim"], n_atoms=N_ATOMS
    ).to(local_device)
    local_net.eval()
    support = torch.linspace(V_MIN, V_MAX, N_ATOMS)
    local_weight_ver = -1
    weight_file = config.get("weight_file", WEIGHT_FILE)
    envs_per_worker = config.get("envs_per_worker", 8)

    def sync_weights():
        nonlocal local_weight_ver
        try:
            ver = weight_version.value
            if ver > local_weight_ver and os.path.exists(weight_file):
                sd = torch.load(weight_file, map_location="cpu", weights_only=True)
                local_net.load_state_dict(sd)
                local_weight_ver = ver
        except Exception:
            pass

    rng = random.Random(worker_id * 10000 + int(time.time()))
    envs = []
    obs_list = []
    nsteps = []
    ep_bufs = []
    for _ in range(envs_per_worker):
        w = DuelArena(seed=rng.randint(0, 2 ** 31))
        o = w.reset()
        envs.append(w)
        obs_list.append(o)
        nsteps.append(NStepBuffer(N_STEP, GAMMA))
        ep_bufs.append({"s": [], "a": [], "r": [], "ns": [], "d": [], "tr": 0.0})
    SEND_EVERY = 5
    pending_trans = []
    pending_results = []

    while not stop_flag.value:
        sync_weights()
        active = [i for i in range(envs_per_worker) if not envs[i].round_over]
        if not active:
            for i in range(envs_per_worker):
                buf = ep_bufs[i]
                if buf["s"]:
                    pending_trans.append((
                        np.array(buf["s"]), np.array(buf["a"]),
                        np.array(buf["r"]), np.array(buf["ns"]), np.array(buf["d"])))
                pending_results.append((envs[i].winner, buf["tr"], envs[i].step_count))
                envs[i] = DuelArena(seed=rng.randint(0, 2 ** 31))
                obs_list[i] = envs[i].reset()
                nsteps[i].reset()
                ep_bufs[i] = {"s": [], "a": [], "r": [], "ns": [], "d": [], "tr": 0.0}
            active = list(range(envs_per_worker))
        batch_obs = np.stack([obs_list[i] for i in active])
        batch_obs_t = torch.from_numpy(batch_obs)
        with torch.no_grad():
            local_net.reset_noise()
            lp = local_net(batch_obs_t)
            probs = lp.exp()
            qv = (probs * support.unsqueeze(0).unsqueeze(0)).sum(-1)
            ai_acts = qv.argmax(dim=-1).numpy()
        for j, i in enumerate(active):
            p_act = rule_based_player(envs[i])
            ai_act = int(ai_acts[j])
            obs2, reward, done = envs[i].step(p_act, ai_act)
            ep_bufs[i]["tr"] += reward
            nsteps[i].push((obs_list[i], ai_act, reward, obs2, float(done)))
            nt = nsteps[i].get()
            if nt:
                ep_bufs[i]["s"].append(nt[0])
                ep_bufs[i]["a"].append(nt[1])
                ep_bufs[i]["r"].append(nt[2])
                ep_bufs[i]["ns"].append(nt[3])
                ep_bufs[i]["d"].append(nt[4])
            obs_list[i] = obs2
            if done:
                for t in nsteps[i].flush():
                    ep_bufs[i]["s"].append(t[0])
                    ep_bufs[i]["a"].append(t[1])
                    ep_bufs[i]["r"].append(t[2])
                    ep_bufs[i]["ns"].append(t[3])
                    ep_bufs[i]["d"].append(t[4])
        for i in range(envs_per_worker):
            if envs[i].round_over:
                buf = ep_bufs[i]
                if buf["s"]:
                    pending_trans.append((
                        np.array(buf["s"]), np.array(buf["a"]),
                        np.array(buf["r"]), np.array(buf["ns"]), np.array(buf["d"])))
                pending_results.append((envs[i].winner, buf["tr"], envs[i].step_count))
                envs[i] = DuelArena(seed=rng.randint(0, 2 ** 31))
                obs_list[i] = envs[i].reset()
                nsteps[i].reset()
                ep_bufs[i] = {"s": [], "a": [], "r": [], "ns": [], "d": [], "tr": 0.0}
        if len(pending_results) >= SEND_EVERY:
            for tr in pending_trans:
                try:
                    transition_queue.put(tr, timeout=0.3)
                except Exception:
                    pass
            pending_trans.clear()
            for r in pending_results:
                try:
                    result_queue.put_nowait(r)
                except Exception:
                    pass
            pending_results.clear()


class ParallelCollector:
    def __init__(self, num_workers, config, net):
        self.num_workers = num_workers
        self.config = config
        self.workers = []
        self.transition_queue = mp.Queue(maxsize=2000)
        self.result_queue = mp.Queue(maxsize=5000)
        self.weight_version = mp.Value("i", 0)
        self.stop_flag = mp.Value("b", False)
        os.makedirs(CKPT_DIR, exist_ok=True)
        self.broadcast_weights(net)

    def broadcast_weights(self, net):
        cpu_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        torch.save(cpu_state, WEIGHT_FILE)
        self.weight_version.value += 1

    def start(self):
        wc = {
            "hidden_dim": self.config["hidden_dim"],
            "weight_file": WEIGHT_FILE,
            "envs_per_worker": self.config.get("envs_per_worker", 8),
        }
        self.stop_flag.value = False
        for i in range(self.num_workers):
            p = mp.Process(
                target=_batched_env_worker,
                args=(i, self.transition_queue, self.result_queue,
                      self.weight_version, self.stop_flag, wc), daemon=True)
            p.start()
            self.workers.append(p)
        te = self.num_workers * wc["envs_per_worker"]
        print(f"  🚀 {self.num_workers} workers × {wc['envs_per_worker']} envs = {te} parallel")

    def collect_and_insert(self, memory, max_batches=500):
        results = []
        total = 0
        n = 0
        while n < max_batches:
            try:
                bd = self.transition_queue.get_nowait()
                s, a, r, ns, d = bd
                memory.push_batch(s, a, r, ns, d)
                total += len(s)
                n += 1
            except Exception:
                break
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except Exception:
                break
        return total, results

    def stop(self):
        self.stop_flag.value = True
        time.sleep(0.5)
        for p in self.workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        self.workers.clear()
        for q in [self.transition_queue, self.result_queue]:
            while True:
                try:
                    q.get_nowait()
                except Exception:
                    break
        print("  ⏹️  Workers stopped")

    def get_queue_size(self):
        try:
            return self.transition_queue.qsize()
        except Exception:
            return 0


# ═══════════════════════════════════════
#  硬件检测
# ═══════════════════════════════════════

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
            elif sys.platform == "darwin":
                import subprocess
                return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) // (1024 * 1024)
        except Exception:
            pass
        return 8192

    def _classify_gpu(self):
        if self.vram_mb >= 10000:
            self.tier = "gpu_xl"
        elif self.vram_mb >= 6000:
            self.tier = "gpu_large"
        elif self.vram_mb >= 4000:
            self.tier = "gpu_medium"
        elif self.vram_mb >= 2000:
            self.tier = "gpu_small"
        else:
            self.tier = "cpu_fallback"
            return
        self.device = torch.device("cuda")

    def _classify_cpu(self):
        if self.ram_mb >= 16000 and self.cpu_cores >= 8:
            self.tier = "cpu_high"
        elif self.ram_mb >= 8000:
            self.tier = "cpu_mid"
        else:
            self.tier = "cpu_low"

    def _build_config(self):
        profiles = {
            "gpu_xl": dict(hidden_dim=512, batch_size=512, memory_size=200000,
                           train_iters=16, use_amp=True, warmup=1000, envs_per_worker=16, wfrac=0.6),
            "gpu_large": dict(hidden_dim=512, batch_size=384, memory_size=150000,
                              train_iters=12, use_amp=True, warmup=800, envs_per_worker=12, wfrac=0.6),
            "gpu_medium": dict(hidden_dim=384, batch_size=256, memory_size=100000,
                               train_iters=8, use_amp=True, warmup=600, envs_per_worker=8, wfrac=0.5),
            "gpu_small": dict(hidden_dim=256, batch_size=128, memory_size=60000,
                              train_iters=6, use_amp=True, warmup=400, envs_per_worker=6, wfrac=0.4),
            "cpu_high": dict(hidden_dim=256, batch_size=128, memory_size=60000,
                             train_iters=4, use_amp=False, warmup=400, envs_per_worker=6, wfrac=0.6),
            "cpu_mid": dict(hidden_dim=192, batch_size=64, memory_size=30000,
                            train_iters=2, use_amp=False, warmup=300, envs_per_worker=4, wfrac=0.5),
            "cpu_low": dict(hidden_dim=128, batch_size=48, memory_size=15000,
                            train_iters=1, use_amp=False, warmup=200, envs_per_worker=2, wfrac=0.4),
            "cpu_fallback": dict(hidden_dim=128, batch_size=32, memory_size=10000,
                                 train_iters=1, use_amp=False, warmup=150, envs_per_worker=2, wfrac=0.3),
        }
        return profiles.get(self.tier, profiles["cpu_low"])

    def _print_report(self):
        print(f"\n{'=' * 60}")
        print(f"  ⚙️  Hardware — {self.tier}")
        print(f"{'=' * 60}")
        if self.has_cuda:
            print(f"  GPU: {self.gpu_name} ({self.vram_mb}MB)")
        else:
            print(f"  GPU: N/A")
        print(f"  CPU: {self.cpu_cores} cores | RAM: {self.ram_mb}MB")
        for k, v in self.config.items():
            print(f"  {k:20s}: {v}")
        print(f"{'=' * 60}\n")


# ═══════════════════════════════════════
#  存档/ 加载
# ═══════════════════════════════════════

def migrate_weights_v4(model, old_sd, label=""):
    ns = model.state_dict()
    matched = 0
    for k in ns:
        if k in old_sd and old_sd[k].shape == ns[k].shape:
            ns[k] = old_sd[k]
            matched += 1
    skipped = len(ns) - matched
    model.load_state_dict(ns)
    tag = "✅" if skipped == 0 else "🔄"
    print(f"  {tag} {label}: {matched}/{len(ns)} matched, {skipped} re-init")


def save_checkpoint(net, target_net, optimizer, scheduler, memory,
                    stats, pool, episode, best_reward, hidden_dim, device):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({
        "version": VERSION, "episode": episode,
        "best_reward": best_reward, "hidden_dim": hidden_dim,
        "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "n_atoms": N_ATOMS,
        "net": net.state_dict(),
        "target": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, CKPT_MODEL)
    ss = {k: (v[-5000:] if isinstance(v, list) else v) for k, v in stats.items()}
    with open(CKPT_STATS, "w") as f:
        json.dump(ss, f, indent=2)
    pool.save(CKPT_POOL)
    print(f"  💾 Saved ep={episode} best_r={best_reward:.1f}")


def save_best(net, best_reward, episode, hidden_dim):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({
        "version": VERSION, "episode": episode,
        "best_reward": best_reward, "hidden_dim": hidden_dim,
        "net": net.state_dict(),
    }, CKPT_BEST)
    print(f"🏆 Best R={best_reward:.1f}")


def load_checkpoint(net, target_net, optimizer, scheduler, pool, hidden_dim, device):
    ds = {"ai_wins": 0, "player_wins": 0, "draws": 0,
          "rewards": [], "winrates": [], "losses": [], "best_streak": 0}
    if not os.path.exists(CKPT_MODEL):
        print("  🆕 No checkpoint")
        return 0, -1e9, ds
    ckpt = torch.load(CKPT_MODEL, map_location=device, weights_only=False)
    oh = ckpt.get("hidden_dim", 128)
    print(f"  📂 v{ckpt.get('version', '?')} hidden={oh}→{hidden_dim}")
    if oh == hidden_dim and ckpt.get("n_atoms", 51) == N_ATOMS:
        try:
            net.load_state_dict(ckpt["net"])
            target_net.load_state_dict(ckpt.get("target", ckpt["net"]))
            print(f"  ✅ Perfect load")
        except Exception:
            migrate_weights_v4(net, ckpt["net"], "net")
            migrate_weights_v4(target_net, ckpt.get("target", ckpt["net"]), "target")
    else:
        migrate_weights_v4(net, ckpt["net"], "net")
        migrate_weights_v4(target_net, ckpt.get("target", ckpt["net"]), "target")
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception:
        print("  ⚠ Optimizer re-init")
    if "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
    pool.load(CKPT_POOL)
    stats = dict(ds)
    if os.path.exists(CKPT_STATS):
        try:
            with open(CKPT_STATS) as f:
                loaded = json.load(f)
            for k in stats:
                if k in loaded:
                    stats[k] = loaded[k]
        except Exception:
            pass
    ep = ckpt.get("episode", 0)
    best = ckpt.get("best_reward", -1e9)
    print(f"  ✅ Resumed ep={ep} best={best:.1f}")
    return ep, best, stats


def load_best_for_play(net, device, hidden_dim):
    """专门为PvAI加载最佳模型"""
    for path in [CKPT_BEST, CKPT_MODEL]:
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
                oh = ckpt.get("hidden_dim", 128)
                if oh == hidden_dim and ckpt.get("n_atoms", 51) == N_ATOMS:
                    net.load_state_dict(ckpt["net"])
                    print(f"  ✅ Loaded AI from {path} (ep={ckpt.get('episode', '?')})")
                else:
                    migrate_weights_v4(net, ckpt["net"], "play")
                return True
            except Exception as e:
                print(f"  ⚠ Load {path} failed: {e}")
    print("  ⚠ No trained model found — AI will use random weights")
    return False


# ═══════════════════════════════════════
#  GUI 组件
# ═══════════════════════════════════════

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
        pygame.draw.rect(surf, (40, 40, 60), (self.x, self.y, self.w, self.h), 1)
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


class QValueBar:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.q_values = np.zeros(ACTION_DIM)

    def update(self, q):
        self.q_values = q

    def draw(self, surf, font):
        import pygame
        pygame.draw.rect(surf, (10, 10, 22), (self.x, self.y, self.w, self.h))
        pygame.draw.rect(surf, (40, 40, 60), (self.x, self.y, self.w, self.h), 1)
        surf.blit(font.render("Q-Values", True, C_DIM), (self.x + 4, self.y + 2))
        if np.all(self.q_values == 0):
            return
        bay = self.y + 15
        bh = self.h - 18
        bw = (self.w - 12) / ACTION_DIM - 2
        qmin, qmax = self.q_values.min(), self.q_values.max()
        qr = qmax - qmin if qmax != qmin else 1.0
        best_a = self.q_values.argmax()
        for i in range(ACTION_DIM):
            bx = self.x + 6 + i * (bw + 2)
            norm = (self.q_values[i] - qmin) / qr
            h_ = max(2, int(norm * (bh - 12)))
            by = bay + bh - h_ - 2
            color = C_GOOD if i == best_a else (60, 60, 90)
            pygame.draw.rect(surf, color, (int(bx), int(by), int(bw), h_))
            txt = font.render(DIR_NAMES[i][0], True, C_DIM)
            surf.blit(txt, (int(bx + bw // 2 - txt.get_width() // 2), int(bay + bh - 12)))


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
        import pygame
        a = max(0.0, self.life / self.max_life)
        r = int(self.size * a)
        if r > 0:
            c = tuple(min(255, int(v * a)) for v in self.color)
            pygame.draw.circle(surf, c, (int(self.x), int(self.y)), r)


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
        self.chart_winrate = MiniChart(px, 10, cw, 58, "AI WinRate%", C_CHART_WIN)
        self.chart_reward = MiniChart(px, 74, cw, 58, "Reward", C_CHART_REW)
        self.chart_eps = MiniChart(px, 138, cw, 58, "Epsilon/Noise", C_CHART_EPS)
        self.chart_loss = MiniChart(px, 202, cw, 58, "Loss", C_CHART_LOSS)
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
                    pygame.draw.rect(self.screen, C_WALL, (rx, ry, CELL, CELL))
                    pygame.draw.rect(self.screen, C_WALL_L, (rx + 1, ry + 1, CELL - 2, CELL - 2), 1)
                elif cell == BRICK:
                    pygame.draw.rect(self.screen, C_BRICK, (rx, ry, CELL, CELL))
                    pygame.draw.rect(self.screen, C_BRICK_D, (rx + 2, ry + 2, CELL - 4, CELL - 4))
                    pygame.draw.line(self.screen, C_BRICK_D, (rx, ry + CELL // 2), (rx + CELL, ry + CELL // 2), 1)
                    pygame.draw.line(self.screen, C_BRICK_D, (rx + CELL // 2, ry), (rx + CELL // 2, ry + CELL), 1)
                else:
                    pygame.draw.rect(self.screen, C_FLOOR, (rx, ry, CELL, CELL))
                    d = dmap[y][x]
                    if d > 0.01:
                        ds = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                        ds.fill((255, 50, 0, int(d * 100)))
                        self.screen.blit(ds, (rx, ry))
                pygame.draw.rect(self.screen, C_GRID, (rx, ry, CELL, CELL), 1)
        pu_icons = ["R", "S", "B"]
        pu_colors = [C_POWERUP, C_WARN, C_SHIELD]
        for pu in world.powerups:
            px_ = pu.x * CELL + CELL // 2
            py_ = pu.y * CELL + CELL // 2
            r = int(CELL * 0.3 + math.sin(self.pulse * 2) * 2)
            c = pu_colors[pu.kind % 3]
            pygame.draw.circle(self.screen, c, (px_, py_), max(r, 4))
            txt = self.fonts["sm"].render(pu_icons[pu.kind % 3], True, (0, 0, 0))
            self.screen.blit(txt, (px_ - txt.get_width() // 2, py_ - txt.get_height() // 2))
        for b in world.bombs:
            bx = b.x * CELL + CELL // 2
            by = b.y * CELL + CELL // 2
            urgency = 1.0 - b.timer / BOMB_TIMER
            r = int(CELL * 0.35 + urgency * 4)
            flash = b.timer <= 3 and b.timer % 2 == 0
            bc = C_BOMB_FUSE if flash else C_BOMB
            pygame.draw.circle(self.screen, bc, (bx, by), max(r, 4))
            pygame.draw.line(self.screen, C_BOMB_FUSE, (bx, by - r), (bx + 3, by - r - 5), 2)
            txt = self.fonts["sm"].render(str(b.timer), True, (0, 0, 0))
            self.screen.blit(txt, (bx - txt.get_width() // 2, by - txt.get_height() // 2))
        for exp in world.explosions:
            ex, ey = exp.x * CELL, exp.y * CELL
            ci = min(exp.timer, len(C_EXPLODE) - 1)
            pygame.draw.rect(self.screen, C_EXPLODE[ci], (ex + 2, ey + 2, CELL - 4, CELL - 4))
        self._draw_fighter(world.player, C_PLAYER, C_PLAYER_D, "P")
        self._draw_fighter(world.ai, C_AI, C_AI_D, "AI")
        i = 0
        while i < len(self.particles):
            if self.particles[i].update(dt):
                self.particles[i].draw(self.screen)
                i += 1
            else:
                self.particles.pop(i)
        pygame.draw.rect(self.screen, (80, 80, 120), (0, 0, ARENA_W, ARENA_H), 3)

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
        self.screen.blit(txt, (fx - txt.get_width() // 2, fy - txt.get_height() // 2))
        hp_w = CELL - 6
        hp_h = 4
        hp_x = fighter.x * CELL + 3
        hp_y = fighter.y * CELL - 6
        pygame.draw.rect(self.screen, (40, 40, 40), (hp_x, hp_y, hp_w, hp_h))
        fill = int(hp_w * fighter.hp / fighter.max_hp)
        hc = C_GOOD if fighter.hp > 1 else C_BAD
        pygame.draw.rect(self.screen, hc, (hp_x, hp_y, fill, hp_h))
        act = fighter.last_action
        if act < 4:
            dx, dy = DIR_MAP[act]
            ax1, ay1 = fx + dx * 8, fy + dy * 8
            ax2, ay2 = fx + dx * 16, fy + dy * 16
            pygame.draw.line(self.screen, c, (ax1, ay1), (ax2, ay2), 2)
            pygame.draw.circle(self.screen, c, (ax2, ay2), 3)
        elif act == 4:
            pygame.draw.circle(self.screen, C_BOMB, (fx, fy - r - 6), 4)

    def draw_panel(self, world, episode, last_loss, mode, speed,
                   ai_wins, player_wins, total_rounds,
                   pool_gen, fps_val, lr_val, streak, best_streak,
                   is_eval, difficulty_label, ai_act_name, q_np):
        import pygame
        px = ARENA_W
        pygame.draw.rect(self.screen, C_PANEL, (px, 0, PANEL_W, ARENA_H))
        pygame.draw.line(self.screen, (50, 50, 80), (px, 0), (px, ARENA_H), 2)
        self.chart_winrate.draw(self.screen, self.fonts["sm"])
        self.chart_reward.draw(self.screen, self.fonts["sm"])
        self.chart_eps.draw(self.screen, self.fonts["sm"])
        self.chart_loss.draw(self.screen, self.fonts["sm"])
        self.qbar.draw(self.screen, self.fonts["sm"])
        iy = 332
        ipx = px + 10
        hp = "♥" * world.player.hp + "·" * (MAX_HP - world.player.hp)
        ha = "♥" * world.ai.hp + "·" * (MAX_HP - world.ai.hp)
        wr = ai_wins / max(total_rounds, 1) * 100
        mode_str = mode
        if is_eval:
            mode_str += " [EVAL]"
        infos = [
            ("Mode", mode_str, C_HIGHLIGHT if mode == "PvAI" else C_GOOD),
            ("Difficulty", difficulty_label, C_WARN),
            ("Round", f"{episode}", C_TEXT),
            ("Step", f"{world.step_count}/{MAX_ROUND_STEPS}", C_TEXT),
            ("P HP", hp, C_PLAYER),
            ("AI HP", ha, C_AI),
            ("AI Act", ai_act_name, C_AI),
            ("", "", C_TEXT),
            ("P Wins", f"{player_wins}", C_PLAYER),
            ("AI Wins", f"{ai_wins}", C_AI),
            ("WinRate", f"{wr:.1f}%", C_WARN),
            ("Streak", f"{streak} (best:{best_streak})", C_GOOD),
            ("", "", C_TEXT),
            ("Loss", f"{last_loss:.5f}", C_TEXT),
            ("LR", f"{lr_val:.6f}", C_DIM),
            ("Gen", f"{pool_gen}", C_GOOD),
            ("Speed", f"x{speed}", C_TEXT),
            ("FPS", f"{fps_val:.0f}", C_DIM),
        ]
        for lbl, val, color in infos:
            if lbl:
                self.screen.blit(self.fonts["sm"].render(f"{lbl}:", True, C_DIM), (ipx, iy))
                self.screen.blit(self.fonts["sm"].render(str(val), True, color), (ipx + 68, iy))
            iy += 13

    def draw_bottom(self, mode, global_step, device_str, difficulty_name):
        import pygame
        by = ARENA_H
        pygame.draw.rect(self.screen, C_BOTTOM, (0, by, WIN_W, BOTTOM_H))
        pygame.draw.line(self.screen, (50, 50, 80), (0, by), (WIN_W, by), 2)
        y1, y2, y3, y4 = by + 6, by + 22, by + 38, by + 54
        self.screen.blit(self.fonts["lg"].render(
            f"Grid Duel Arena v{VERSION} [{self.hw_tier}]", True, C_HIGHLIGHT), (10, y1))
        self.screen.blit(self.fonts["sm"].render(
            "[WASD]Move [J]Bomb [K]Stay | [1]PvAI [2]SelfPlay [3]Train", True, C_DIM), (10, y2))
        self.screen.blit(self.fonts["sm"].render(
            "[Space]Pause [Up/Dn]Speed [F1-F4]Difficulty [Ctrl+S]Save [Esc]Quit", True, C_DIM), (10, y3))
        self.screen.blit(self.fonts["sm"].render(
            f"Mode:{mode} | AI:{difficulty_name} | {device_str}", True, C_DIM), (10, y4))

    def draw_round_result(self, winner, p_wins, ai_wins, total, difficulty_label):
        import pygame
        ov = pygame.Surface((ARENA_W, ARENA_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 160))
        self.screen.blit(ov, (0, 0))
        cx, cy = ARENA_W // 2, ARENA_H // 2
        if winner == "player":
            txt, color = "YOU WIN!", C_PLAYER
        elif winner == "ai":
            txt, color = "AI WINS!", C_AI
        else:
            txt, color = "DRAW!", C_HIGHLIGHT
        title = self.fonts["xl"].render(txt, True, color)
        self.screen.blit(title, (cx - title.get_width() // 2, cy - 50))
        score = self.fonts["md"].render(
            f"You {p_wins} : {ai_wins} AI(of {total})  {difficulty_label}", True, C_TEXT)
        self.screen.blit(score, (cx - score.get_width() // 2, cy + 10))
        hint = self.fonts["sm"].render("[N] Next Round  |  [F1-F4] Difficulty  |  [Esc] Quit", True, C_DIM)
        self.screen.blit(hint, (cx - hint.get_width() // 2, cy + 40))


# ═══════════════════════════════════════
#  命令行
# ═══════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=f"Grid Duel Arena v{VERSION}")
    p.add_argument("--headless", action="store_true", help="纯训练模式")
    p.add_argument("--gui", action="store_true", help="强制GUI")
    p.add_argument("--play", action="store_true", help="纯PvAI对战模式(不训练)")
    p.add_argument("--episodes", type=int, default=0)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--lr", type=float, default=0)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--hidden", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--difficulty", type=str, default="Normal",
                   choices=DIFFICULTY_ORDER, help="AI难度")
    return p.parse_args()


def detect_display():
    if sys.platform == "linux":
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return False
    try:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        pygame.init()
        info = pygame.display.Info()
        ok = info.current_w > 0 and info.current_h > 0
        pygame.quit()
        return ok
    except Exception:
        return False


# ═══════════════════════════════════════
#  Headless 训练主循环
# ═══════════════════════════════════════

def main_headless():
    args = parse_args()
    hw = HardwareProfile(forced_device=args.device)
    DEVICE = hw.device
    CFG = hw.config
    HIDDEN = args.hidden if args.hidden > 0 else CFG["hidden_dim"]
    BS = args.batch_size if args.batch_size > 0 else CFG["batch_size"]
    MS = CFG["memory_size"]
    USE_AMP = CFG["use_amp"]
    WARMUP = CFG["warmup"]
    TI = CFG["train_iters"]
    nw = args.workers if args.workers > 0 else max(2, int(hw.cpu_cores * CFG["wfrac"]))
    te = nw * CFG["envs_per_worker"]

    print(f"{'=' * 65}")
    print(f"🚀 RAINBOW DQN v{VERSION} — Headless Training")
    print(f"  Network: hidden={HIDDEN}, atoms={N_ATOMS}, {N_STEP}-step")
    print(f"  Parallel: {nw} workers × {CFG['envs_per_worker']} = {te} envs")
    print(f"  Batch={BS}, Memory={MS}, AMP={USE_AMP}")
    print(f"{'=' * 65}")

    net = RainbowNet(hidden=HIDDEN, n_atoms=N_ATOMS).to(DEVICE)
    target_net = RainbowNet(hidden=HIDDEN, n_atoms=N_ATOMS).to(DEVICE)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()
    support = torch.linspace(V_MIN, V_MAX, N_ATOMS).to(DEVICE)
    delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)
    gamma_n = GAMMA ** N_STEP
    lr = args.lr if args.lr > 0 else LR
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5, eps=1.5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3000, T_mult=2, eta_min=LR_MIN)
    memory = SumTreePER(MS, STATE_DIM)
    pool = StrategyPool()
    amp_ctx = AMPContext(USE_AMP, DEVICE)
    start_ep, best_reward, stats = load_checkpoint(
        net, target_net, optimizer, scheduler, pool, HIDDEN, DEVICE)
    pw = stats.get("player_wins", 0)
    aw = stats.get("ai_wins", 0)
    dw = stats.get("draws", 0)
    tr = pw + aw + dw
    ai_stk = 0
    best_stk = stats.get("best_streak", 0)
    gs = start_ep * 150
    ll = 0.0
    ep = start_ep
    ema_loss = 0.0
    si = args.save_interval
    me = args.episodes

    collector = ParallelCollector(nw, CFG, net)
    collector.start()
    t0 = time.time()
    tst = 0
    lr_t = time.time()

    print(f"\n📊 ep={start_ep}, Ctrl+C to stop\n")
    try:
        while True:
            if me > 0 and ep >= start_ep + me:
                break
            ti, results = collector.collect_and_insert(memory, max_batches=500)
            for r in results:
                winner, trew, steps = r
                tr += 1
                if winner == "ai":
                    aw += 1
                    ai_stk += 1
                    best_stk = max(best_stk, ai_stk)
                elif winner == "player":
                    pw += 1
                    ai_stk = 0
                else:
                    dw += 1
                    ai_stk = 0
                wr = aw / max(tr, 1) * 100
                stats.setdefault("rewards", []).append(trew)
                stats.setdefault("winrates", []).append(wr)
                stats.setdefault("losses", []).append(ll)
                if trew > best_reward:
                    best_reward = trew
                    save_best(net, best_reward, ep, HIDDEN)
                ep += 1
                if ep > 0 and ep % 25 == 0:
                    rr = stats.get("rewards", [])[-25:]
                    if rr:
                        pool.add(f"g{pool.generation}", net.state_dict(), sum(rr) / len(rr))
                if ep % si == 0:
                    stats.update({"player_wins": pw, "ai_wins": aw, "draws": dw, "best_streak": best_stk})
                    save_checkpoint(net, target_net, optimizer, scheduler, memory, stats, pool, ep, best_reward, HIDDEN, DEVICE)
            can_train = len(memory) >= max(BS, WARMUP)
            if can_train and ti > 0:
                iters = max(TI, ti // (BS // 4))
                iters = min(iters, 200)
                for _ in range(iters):
                    if len(memory) < BS:
                        break
                    net.reset_noise()
                    target_net.reset_noise()
                    pb = min(1.0, 0.4 + gs * 0.00005)
                    batch, tidx, isw = memory.sample(BS, pb)
                    optimizer.zero_grad(set_to_none=True)
                    with amp_ctx.autocast():
                        loss, td_err = compute_c51_loss(net, target_net, batch, support, delta_z, DEVICE, isw, gamma_n)
                    amp_ctx.scale_and_step(loss, optimizer, net.parameters(), 10.0)
                    memory.update_priorities(tidx, td_err)
                    ll = loss.item()
                    ema_loss = 0.99 * ema_loss + 0.01 * ll
                    tst += 1
                    gs += 1
                    for tp, sp in zip(target_net.parameters(), net.parameters()):
                        tp.data.mul_(1 - TAU).add_(sp.data, alpha=TAU)
                    scheduler.step()
            if tst > 0 and tst % 20 == 0:
                collector.broadcast_weights(net)
            now = time.time()
            if now - lr_t >= 5.0:
                el = now - t0
                ed = ep - start_ep
                epm = ed / (el / 60) if el > 0 else 0
                wr = aw / max(tr, 1) * 100
                qs = collector.get_queue_size()
                rr = stats.get("rewards", [-1])[-50:]
                ar = sum(rr) / max(len(rr), 1)
                clr = optimizer.param_groups[0]["lr"]
                gi = ""
                if DEVICE.type == "cuda":
                    try:
                        mu = torch.cuda.memory_allocated() / (1024 ** 2)
                        gi = f"GPU:{mu:.0f}MB"
                    except Exception:
                        pass
                print(f"EP{ep:6d}|WR:{wr:5.1f}%|R:{ar:6.1f}|L:{ema_loss:.4f}|lr:{clr:.1e}|"
                      f"Stk:{ai_stk}/{best_stk}|Mem:{len(memory):6d}|Q:{qs:3d}|"
                      f"T:{tst:7d}|{epm:5.1f}ep/m|{el / 3600:.2f}h|{gi}")
                lr_t = now
            if not ti and not results:
                time.sleep(0.003)
    except KeyboardInterrupt:
        print("\n  ⏹️  Interrupted")
    collector.stop()
    stats.update({"player_wins": pw, "ai_wins": aw, "draws": dw, "best_streak": best_stk})
    save_checkpoint(net, target_net, optimizer, scheduler, memory, stats, pool, ep, best_reward, HIDDEN, DEVICE)
    el = time.time() - t0
    ed = ep - start_ep
    wr = aw / max(tr, 1) * 100
    print(f"\n{'=' * 65}")
    print(f"  📊 Done: {ed} eps, {el / 3600:.2f}h, WR={wr:.1f}%, best_stk={best_stk}")
    print(f"{'=' * 65}\n")


# ═══════════════════════════════════════
#  GUI 主循环 — 完整PvAI + 训练
# ═══════════════════════════════════════

def main_gui():
    args = parse_args()
    hw = HardwareProfile(forced_device=args.device)
    DEVICE = hw.device
    CFG = hw.config
    HIDDEN = args.hidden if args.hidden > 0 else min(CFG["hidden_dim"], 512)
    BS = args.batch_size if args.batch_size > 0 else CFG["batch_size"]
    MS = CFG["memory_size"]
    USE_AMP = CFG["use_amp"]
    WARMUP = CFG["warmup"]
    PLAY_ONLY = args.play

    import pygame
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"Grid Duel Arena v{VERSION} — Rainbow DQN")
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

    net = RainbowNet(hidden=HIDDEN, n_atoms=N_ATOMS).to(DEVICE)
    target_net = RainbowNet(hidden=HIDDEN, n_atoms=N_ATOMS).to(DEVICE)
    target_net.load_state_dict(net.state_dict())
    target_net.eval()
    support = torch.linspace(V_MIN, V_MAX, N_ATOMS).to(DEVICE)
    delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)
    gamma_n = GAMMA ** N_STEP
    optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5, eps=1.5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3000, T_mult=2, eta_min=LR_MIN)
    memory = SumTreePER(MS, STATE_DIM)
    nstep = NStepBuffer(N_STEP, GAMMA)
    pool = StrategyPool()
    amp_ctx = AMPContext(USE_AMP, DEVICE)
    start_ep, best_reward, stats = load_checkpoint(
        net, target_net, optimizer, scheduler, pool, HIDDEN, DEVICE)

    if PLAY_ONLY:
        load_best_for_play(net, DEVICE, HIDDEN)
        net.eval()

    renderer = Renderer(screen, clock, fonts, hw.tier)
    for v in stats.get("winrates", [])[-150:]:
        renderer.chart_winrate.add(v)
    for v in stats.get("rewards", [])[-150:]:
        renderer.chart_reward.add(v)
    for v in stats.get("losses", [])[-150:]:
        renderer.chart_loss.add(v)

    dev_str = str(DEVICE)
    if hw.has_cuda:
        dev_str += f" ({hw.gpu_name[:25]})"

    # 状态
    mode = "PvAI" if PLAY_ONLY else "PvAI"
    speed = 1
    paused = False
    gs = start_ep * 150
    ll = 0.0
    pw = stats.get("player_wins", 0)
    aw = stats.get("ai_wins", 0)
    dw = stats.get("draws", 0)
    tr = pw + aw + dw
    player_act_q = 5
    ai_stk = 0
    best_stk = stats.get("best_streak", 0)
    episode = start_ep
    difficulty_idx = DIFFICULTY_ORDER.index(args.difficulty)
    difficulty_name = DIFFICULTY_ORDER[difficulty_idx]
    difficulty_cfg = DIFFICULTY_LEVELS[difficulty_name]
    ai_act_name = "—"
    last_q_np = np.zeros(ACTION_DIM)

    print(f"\n  🎮 Grid Duel Arena v{VERSION}")
    print(f"     Device: {DEVICE} | Mode: {'Play Only' if PLAY_ONLY else 'Full'}")
    print(f"     Difficulty: {difficulty_cfg['label']}")
    print(f"     [1]PvAI [2]SelfPlay [3]Train [F1-F4]Difficulty\n")

    running = True
    while running:
        world = DuelArena(seed=random.randint(0, 2 ** 31))
        obs = world.reset()
        total_reward = 0.0
        nstep.reset()
        show_result = False
        result_timer = 0
        is_eval = (episode % EVAL_INTERVAL == 0 and episode > 0 and mode != "PvAI")

        round_running = True
        while round_running and running:
            # 事件
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
                    elif ev.key == pygame.K_2 and not PLAY_ONLY:
                        mode = "SelfPlay"
                        speed = 3
                    elif ev.key == pygame.K_3 and not PLAY_ONLY:
                        mode = "Train"
                        speed = max(speed, 10)
                    elif ev.key == pygame.K_n and show_result:
                        round_running = False
                        continue
                    elif ev.key == pygame.K_F1:
                        difficulty_idx = 0
                    elif ev.key == pygame.K_F2:
                        difficulty_idx = 1
                    elif ev.key == pygame.K_F3:
                        difficulty_idx = 2
                    elif ev.key == pygame.K_F4:
                        difficulty_idx = 3
                    elif (ev.key == pygame.K_s and
                          (pygame.key.get_mods() & pygame.KMOD_CTRL) and
                          not show_result and not PLAY_ONLY):
                        stats.update({"player_wins": pw, "ai_wins": aw, "draws": dw, "best_streak": best_stk})
                        save_checkpoint(net, target_net, optimizer, scheduler,
                                        memory, stats, pool, episode, best_reward, HIDDEN, DEVICE)
                    # PvAI 控制
                    if mode == "PvAI" and not show_result:
                        if ev.key == pygame.K_w:
                            player_act_q = 0
                        elif ev.key == pygame.K_s and not (pygame.key.get_mods() & pygame.KMOD_CTRL):
                            player_act_q = 1
                        elif ev.key == pygame.K_a:
                            player_act_q = 2
                        elif ev.key == pygame.K_d:
                            player_act_q = 3
                        elif ev.key == pygame.K_j:
                            player_act_q = 4
                        elif ev.key == pygame.K_k:
                            player_act_q = 5

            if not running:
                break

            difficulty_name = DIFFICULTY_ORDER[difficulty_idx]
            difficulty_cfg = DIFFICULTY_LEVELS[difficulty_name]

            # 按键持续检测
            if mode == "PvAI" and not show_result and not paused:
                keys = pygame.key.get_pressed()
                ctrl = bool(pygame.key.get_mods() & pygame.KMOD_CTRL)
                if keys[pygame.K_w]:
                    player_act_q = 0
                elif keys[pygame.K_s] and not ctrl:
                    player_act_q = 1
                elif keys[pygame.K_a]:
                    player_act_q = 2
                elif keys[pygame.K_d]:
                    player_act_q = 3

            fps_val = clock.get_fps()

            if paused and not show_result:
                renderer.draw_arena(world)
                renderer.draw_panel(world, episode, ll, mode, speed, aw, pw, tr,
                                    pool.generation, fps_val, optimizer.param_groups[0]["lr"],
                                    ai_stk, best_stk, is_eval, difficulty_cfg["label"],
                                    ai_act_name, last_q_np)
                renderer.draw_bottom(mode, gs, dev_str, difficulty_name)
                ptxt = fonts["lg"].render("PAUSED (Space to resume)", True, C_WARN)
                screen.blit(ptxt, (ARENA_W // 2 - ptxt.get_width() // 2, ARENA_H // 2 - ptxt.get_height() // 2))
                pygame.display.flip()
                clock.tick(15)
                continue

            if show_result:
                renderer.draw_arena(world)
                renderer.draw_panel(world, episode, ll, mode, speed, aw, pw, tr,
                                    pool.generation, fps_val, optimizer.param_groups[0]["lr"],
                                    ai_stk, best_stk, is_eval, difficulty_cfg["label"],
                                    ai_act_name, last_q_np)
                renderer.draw_bottom(mode, gs, dev_str, difficulty_name)
                renderer.draw_round_result(world.winner, pw, aw, tr, difficulty_cfg["label"])
                pygame.display.flip()
                clock.tick(15)
                if mode in ("SelfPlay", "Train"):
                    result_timer += 1
                    if result_timer > (2 if mode == "Train" else 12):
                        round_running = False
                continue

            # 玩家动作
            if mode == "PvAI":
                p_act = player_act_q
                player_act_q = 5
            else:
                p_act = rule_based_player(world)

            # AI 动作
            net.reset_noise()
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                q_vals = net.get_q_values(obs_t, support)
                last_q_np = q_vals.squeeze(0).cpu().numpy()
            renderer.qbar.update(last_q_np)

            # 难度控制
            noise_scale = difficulty_cfg["noise_scale"]
            rand_prob = difficulty_cfg["random_prob"]
            if random.random() < rand_prob:
                ai_act = random.randint(0, ACTION_DIM - 1)
            elif noise_scale > 0:
                noisy_q = last_q_np + np.random.randn(ACTION_DIM) * noise_scale
                ai_act = int(np.argmax(noisy_q))
            else:
                ai_act = q_vals.argmax(dim=-1).item()

            ai_act_name = DIR_NAMES[ai_act]
            prev_ai_hp = world.ai.hp
            prev_pl_hp = world.player.hp
            obs2, reward, done = world.step(p_act, ai_act)
            total_reward += reward
            gs += 1

            for exp in world.new_explosions:
                renderer.add_explosion_particles(exp.x, exp.y)
            if world.ai.hp < prev_ai_hp:
                renderer.add_hit_particles(world.ai.x, world.ai.y, C_AI)
            if world.player.hp < prev_pl_hp:
                renderer.add_hit_particles(world.player.x, world.player.y, C_PLAYER)

            if not is_eval and not PLAY_ONLY:
                nstep.push((obs, ai_act, reward, obs2, float(done)))
                nt = nstep.get()
                if nt:
                    memory.push(nt)
            obs = obs2

            # 训练
            can_train = (len(memory) >= max(BS, WARMUP) and not is_eval and not PLAY_ONLY)
            if can_train and not done:
                net.reset_noise()
                target_net.reset_noise()
                pb = min(1.0, 0.4 + gs * 0.00005)
                batch, tidx, isw = memory.sample(BS, pb)
                optimizer.zero_grad(set_to_none=True)
                with amp_ctx.autocast():
                    loss, td_err = compute_c51_loss(
                        net, target_net, batch, support, delta_z, DEVICE, isw, gamma_n)
                amp_ctx.scale_and_step(loss, optimizer, net.parameters(), 10.0)
                memory.update_priorities(tidx, td_err)
                ll = loss.item()
                for tp, sp in zip(target_net.parameters(), net.parameters()):
                    tp.data.mul_(1 - TAU).add_(sp.data, alpha=TAU)
                scheduler.step()

            # 渲染
            do_render = not (mode == "Train" and gs % 4 != 0)
            if do_render:
                renderer.draw_arena(world)
                renderer.draw_panel(world, episode, ll, mode, speed, aw, pw, tr,
                                    pool.generation, fps_val, optimizer.param_groups[0]["lr"],
                                    ai_stk, best_stk, is_eval, difficulty_cfg["label"],
                                    ai_act_name, last_q_np)
                renderer.draw_bottom(mode, gs, dev_str, difficulty_name)
                pygame.display.flip()

            fps = BASE_FPS * speed
            if mode == "Train":
                fps = max(fps, 300)
            clock.tick(fps)

            if done:
                if not is_eval and not PLAY_ONLY:
                    for t in nstep.flush():
                        memory.push(t)
                show_result = True
                result_timer = 0
                tr += 1
                if world.winner == "ai":
                    aw += 1
                    ai_stk += 1
                    best_stk = max(best_stk, ai_stk)
                elif world.winner == "player":
                    pw += 1
                    ai_stk = 0
                else:
                    dw += 1
                    ai_stk = 0
                wr = aw / max(tr, 1) * 100
                stats.setdefault("rewards", []).append(total_reward)
                stats.setdefault("winrates", []).append(wr)
                stats.setdefault("losses", []).append(ll)
                renderer.chart_winrate.add(wr)
                renderer.chart_reward.add(total_reward)
                renderer.chart_eps.add(difficulty_idx / 3.0)
                renderer.chart_loss.add(ll)
                ws = {"ai": "AI WIN", "player": "P WIN", "draw": "DRAW"}.get(world.winner, "?")
                print(f"EP{episode:5d}|{ws:>6s}|R:{total_reward:7.1f}|"
                      f"AI:{aw} P:{pw}|WR:{wr:5.1f}%|"
                      f"Diff:{difficulty_name}|Stk:{ai_stk}|Mem:{len(memory)}")
                if not PLAY_ONLY:
                    if total_reward > best_reward:
                        best_reward = total_reward
                        save_best(net, best_reward, episode, HIDDEN)
                    if episode > 0 and episode % 25 == 0 and not is_eval:
                        rr = stats["rewards"][-25:]
                        pool.add(f"g{pool.generation}", net.state_dict(), sum(rr) / max(len(rr), 1))
                    if (episode + 1) % 50 == 0:
                        stats.update({"player_wins": pw, "ai_wins": aw,
                                      "draws": dw, "best_streak": best_stk})
                        save_checkpoint(net, target_net, optimizer, scheduler,
                                        memory, stats, pool, episode + 1,
                                        best_reward, HIDDEN, DEVICE)
                episode += 1

        if not PLAY_ONLY:
            stats.update({"player_wins": pw, "ai_wins": aw, "draws": dw, "best_streak": best_stk})
            save_checkpoint(net, target_net, optimizer, scheduler,
                            memory, stats, pool, episode, best_reward, HIDDEN, DEVICE)
    pygame.quit()
    print("\n  👋 Game saved. Goodbye!")


# ═══════════════════════════════════════
#  入口
# ═══════════════════════════════════════

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    if args.headless:
        headless = True
    elif args.gui or args.play:
        headless = False
    else:
        headless = not detect_display()

    if headless:
        print(f"\n  🖥️  Headless Rainbow Training v{VERSION}")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        main_headless()
    else:
        print(f"\n  🎮  GUI Mode v{VERSION}")
        main_gui()