from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIONS = ["up", "down", "left", "right"]
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COORD_BIT_WEIGHTS = (1, 2, 4)
NATURALISTIC_NORMAL_FAMILIES = ("open_room", "wall_detour")
PRESSURE_FAMILIES = ("pressure_plate_open_room", "pressure_plate_wall_detour")
TRAP_FAMILIES = ("trap_open_room",)
NORMAL_FAMILIES = NATURALISTIC_NORMAL_FAMILIES + PRESSURE_FAMILIES + TRAP_FAMILIES
CHALLENGE_FAMILIES = ("bridge_shortcut", "key_door", "one_way_fork", "slip_zone", "decoy_reward")
ALL_FAMILIES = NORMAL_FAMILIES + CHALLENGE_FAMILIES
POLICY_FAMILIES = NATURALISTIC_NORMAL_FAMILIES + PRESSURE_FAMILIES + CHALLENGE_FAMILIES
POLICY_FAMILY_ALIAS = {
    "trap_open_room": "open_room",
}
POLICY_FAMILY_TO_INDEX = {family: i for i, family in enumerate(POLICY_FAMILIES)}
STATE_MODES = 4
MAX_TRANSITION_BRANCHING = len(ACTIONS)


@dataclass(frozen=True)
class FamilySlice:
    family_id: str
    count: int
    severity_min: float
    severity_max: float
    slip_value: float = 0.0
    wind_dir: int = 0
    wind_strength: float = 0.0
    dynamics_id: int = -1


@dataclass
class GridworldGeneratorConfig:
    rows: int = 8
    cols: int = 8
    horizon: int = 20
    gamma: float = 1.0
    step_cost: float = -0.01
    goal_reward: float = 1.0
    catastrophe_penalty: float = -1.0
    min_start_goal_distance: int = 7
    min_wall_density: float = 0.04
    max_wall_density: float = 0.18
    max_generation_attempts_per_task: int = 200


@dataclass
class GridworldLayout:
    family_id: str
    subfamily_id: str
    severity: float
    backbone_seed: int
    overlay_seed: int
    rows: int
    cols: int
    horizon: int
    gamma: float
    step_cost: float
    goal_reward: float
    catastrophe_penalty: float
    hazard_terminal: bool
    decoy_reward: float
    decoy_terminal: bool
    goal_requires_unlock: bool
    x_unlock_active: torch.Tensor
    x_unlock_value: torch.Tensor
    y_unlock_active: torch.Tensor
    y_unlock_value: torch.Tensor
    pressure_plate_mask: torch.Tensor
    active_constraint_count: int
    start_pos: tuple[int, int]
    wall_mask: torch.Tensor
    goal_mask: torch.Tensor
    hazard_mask: torch.Tensor
    key_mask: torch.Tensor
    door_mask: torch.Tensor
    decoy_mask: torch.Tensor
    one_way_mask: torch.Tensor
    slip_prob: torch.Tensor
    # Manifest-level drift: with prob wind_strength the executed action is
    # wind_dir regardless of the intended action. 0.0 = no wind (legacy).
    wind_dir: int = 0
    wind_strength: float = 0.0


@dataclass
class GridworldTask:
    task_id: str
    split_name: str
    family_id: str
    subfamily_id: str
    severity: float
    backbone_seed: int
    overlay_seed: int
    canonical_hash: str
    rows: int
    cols: int
    horizon: int
    gamma: float
    step_cost: float
    goal_reward: float
    catastrophe_penalty: float
    hazard_terminal: bool
    decoy_reward: float
    decoy_terminal: bool
    goal_requires_unlock: bool
    x_unlock_active: torch.Tensor
    x_unlock_value: torch.Tensor
    y_unlock_active: torch.Tensor
    y_unlock_value: torch.Tensor
    pressure_plate_mask: torch.Tensor
    active_constraint_count: int
    start_pos: tuple[int, int]
    wall_mask: torch.Tensor
    goal_mask: torch.Tensor
    hazard_mask: torch.Tensor
    key_mask: torch.Tensor
    door_mask: torch.Tensor
    decoy_mask: torch.Tensor
    one_way_mask: torch.Tensor
    slip_prob: torch.Tensor
    feature_tensor: torch.Tensor
    task_vector: torch.Tensor
    transition_next_states: torch.Tensor
    transition_probs: torch.Tensor
    transition_rewards: torch.Tensor
    start_state_index: int
    optimal_value: float
    wind_dir: int = 0
    wind_strength: float = 0.0

    @property
    def num_states(self) -> int:
        return int(STATE_MODES * self.rows * self.cols)

    def summary(self) -> dict:
        return {
            "task_id": self.task_id,
            "split_name": self.split_name,
            "family_id": self.family_id,
            "subfamily_id": self.subfamily_id,
            "severity": float(self.severity),
            "backbone_seed": int(self.backbone_seed),
            "overlay_seed": int(self.overlay_seed),
            "canonical_hash": self.canonical_hash,
            "optimal_value": float(self.optimal_value),
            "active_constraint_count": int(self.active_constraint_count),
        }


@dataclass(frozen=True)
class GridworldTaskSpec:
    family_id: str
    subfamily_id: str
    severity: float
    backbone_seed: int
    overlay_seed: int
    canonical_hash: str
    optimal_value: float

    def summary(self) -> dict:
        return {
            "family_id": self.family_id,
            "subfamily_id": self.subfamily_id,
            "severity": float(self.severity),
            "backbone_seed": int(self.backbone_seed),
            "overlay_seed": int(self.overlay_seed),
            "canonical_hash": self.canonical_hash,
            "optimal_value": float(self.optimal_value),
        }

    def to_json(self) -> dict:
        return self.summary()

    @classmethod
    def from_json(cls, payload: dict) -> "GridworldTaskSpec":
        return cls(
            family_id=str(payload["family_id"]),
            subfamily_id=str(payload.get("subfamily_id", payload["family_id"])),
            severity=float(payload["severity"]),
            backbone_seed=int(payload["backbone_seed"]),
            overlay_seed=int(payload["overlay_seed"]),
            canonical_hash=str(payload["canonical_hash"]),
            optimal_value=float(payload["optimal_value"]),
        )


@dataclass(frozen=True)
class TrapLayoutRecord:
    family_id: str
    subfamily_id: str
    severity: float
    canonical_hash: str
    optimal_value: float
    start_pos: tuple[int, int]
    goal_pos: tuple[int, int]
    hazard_bits: str
    backbone_seed: int
    overlay_seed: int
    slip_value: float = 0.0
    wind_dir: int = 0
    wind_strength: float = 0.0
    # Identifies which dynamics group (slip, wind) this record belongs to in a
    # dynamics-grouped bank; -1 = ungrouped/legacy.
    dynamics_id: int = -1

    def summary(self) -> dict:
        return {
            "record_type": "trap_layout",
            "family_id": self.family_id,
            "subfamily_id": self.subfamily_id,
            "severity": float(self.severity),
            "canonical_hash": self.canonical_hash,
            "optimal_value": float(self.optimal_value),
            "start_pos": [int(self.start_pos[0]), int(self.start_pos[1])],
            "goal_pos": [int(self.goal_pos[0]), int(self.goal_pos[1])],
            "hazard_bits": self.hazard_bits,
            "backbone_seed": int(self.backbone_seed),
            "overlay_seed": int(self.overlay_seed),
            "slip_value": float(self.slip_value),
            "wind_dir": int(self.wind_dir),
            "wind_strength": float(self.wind_strength),
            "dynamics_id": int(self.dynamics_id),
        }

    def to_json(self) -> dict:
        return self.summary()

    @classmethod
    def from_json(cls, payload: dict) -> "TrapLayoutRecord":
        return cls(
            family_id=str(payload["family_id"]),
            subfamily_id=str(payload.get("subfamily_id", payload["family_id"])),
            severity=float(payload["severity"]),
            canonical_hash=str(payload["canonical_hash"]),
            optimal_value=float(payload["optimal_value"]),
            start_pos=(int(payload["start_pos"][0]), int(payload["start_pos"][1])),
            goal_pos=(int(payload["goal_pos"][0]), int(payload["goal_pos"][1])),
            hazard_bits=str(payload["hazard_bits"]),
            backbone_seed=int(payload.get("backbone_seed", 0)),
            overlay_seed=int(payload.get("overlay_seed", 0)),
            slip_value=float(payload.get("slip_value", 0.0)),
            wind_dir=int(payload.get("wind_dir", 0)),
            wind_strength=float(payload.get("wind_strength", 0.0)),
            dynamics_id=int(payload.get("dynamics_id", -1)),
        )


@dataclass
class TaskSplitManifest:
    pretrain_train: list[GridworldTask]
    meta_fit: list[GridworldTask]
    meta_deploy: list[GridworldTask]
    heldout_fit: list[GridworldTask]
    heldout_deploy: list[GridworldTask]

    def by_name(self) -> dict[str, list[GridworldTask]]:
        return {
            "pretrain_train": self.pretrain_train,
            "meta_fit": self.meta_fit,
            "meta_deploy": self.meta_deploy,
            "heldout_fit": self.heldout_fit,
            "heldout_deploy": self.heldout_deploy,
        }

    def summary(self) -> dict:
        out: dict[str, dict[str, object]] = {}
        for name, tasks in self.by_name().items():
            family_counts: dict[str, int] = {}
            for task in tasks:
                family_counts[task.family_id] = family_counts.get(task.family_id, 0) + 1
            out[name] = {
                "count": len(tasks),
                "family_counts": family_counts,
            }
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DensePolicyNet(nn.Module):
    """Small U-net-style dense policy that emits logits for every state."""

    def __init__(
        self,
        in_channels: int,
        task_dim: int,
        rows: int,
        cols: int,
        horizon: int,
        *,
        hidden_channels: int = 32,
        n_actions: int = 4,
        state_modes: int = STATE_MODES,
        task_hidden_dim: int = 64,
    ):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.horizon = horizon
        self.n_actions = n_actions
        self.state_modes = state_modes
        self.task_dim = task_dim
        self.enc1 = ConvBlock(in_channels, hidden_channels)
        self.enc2 = ConvBlock(hidden_channels, hidden_channels * 2)
        self.bottleneck = ConvBlock(hidden_channels * 2, hidden_channels * 2)
        self.dec2 = ConvBlock(hidden_channels * 4, hidden_channels * 2)
        self.dec1 = ConvBlock(hidden_channels * 3, hidden_channels)
        self.task_mlp = nn.Sequential(
            nn.Linear(task_dim, task_hidden_dim),
            nn.GELU(),
            nn.Linear(task_hidden_dim, task_hidden_dim),
            nn.GELU(),
        )
        self.enc1_film = nn.Linear(task_hidden_dim, 2 * hidden_channels)
        self.enc2_film = nn.Linear(task_hidden_dim, 4 * hidden_channels)
        self.bottleneck_film = nn.Linear(task_hidden_dim, 4 * hidden_channels)
        self.dec2_film = nn.Linear(task_hidden_dim, 4 * hidden_channels)
        self.dec1_film = nn.Linear(task_hidden_dim, 2 * hidden_channels)
        out_channels = (horizon + 1) * state_modes * n_actions
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        for layer in (
            self.enc1_film,
            self.enc2_film,
            self.bottleneck_film,
            self.dec2_film,
            self.dec1_film,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _apply_film(self, x: torch.Tensor, task_embed: torch.Tensor, film_layer: nn.Linear) -> torch.Tensor:
        gamma, beta = film_layer(task_embed).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta

    def forward(self, features: torch.Tensor, task_vectors: torch.Tensor) -> torch.Tensor:
        task_embed = self.task_mlp(task_vectors)
        x1 = self._apply_film(self.enc1(features), task_embed, self.enc1_film)
        x2 = self._apply_film(self.enc2(F.avg_pool2d(x1, kernel_size=2)), task_embed, self.enc2_film)
        xb = self._apply_film(self.bottleneck(F.avg_pool2d(x2, kernel_size=2)), task_embed, self.bottleneck_film)
        u2 = F.interpolate(xb, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self._apply_film(self.dec2(torch.cat([u2, x2], dim=1)), task_embed, self.dec2_film)
        u1 = F.interpolate(u2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self._apply_film(self.dec1(torch.cat([u1, x1], dim=1)), task_embed, self.dec1_film)
        logits = self.head(u1)
        batch = logits.shape[0]
        return logits.view(batch, self.horizon + 1, self.state_modes, self.n_actions, self.rows, self.cols)


def state_index(has_key: int, reward_unlocked: int, y: int, x: int, rows: int, cols: int) -> int:
    mode = 2 * int(has_key) + int(reward_unlocked)
    return mode * rows * cols + y * cols + x


def decode_state(index: int, rows: int, cols: int) -> tuple[int, int, int, int]:
    per_key = rows * cols
    mode = index // per_key
    rem = index % per_key
    has_key = mode // 2
    reward_unlocked = mode % 2
    y = rem // cols
    x = rem % cols
    return has_key, reward_unlocked, y, x


def dense_logits_to_state_action_logits(
    dense_logits: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    return dense_logits.permute(0, 1, 2, 4, 5, 3).reshape(
        dense_logits.shape[0],
        dense_logits.shape[1],
        dense_logits.shape[2] * rows * cols,
        dense_logits.shape[3],
    )


def _empty_grid(rows: int, cols: int) -> torch.Tensor:
    return torch.zeros(rows, cols, dtype=torch.float32)


def _pack_bool_mask_hex(mask: torch.Tensor) -> str:
    flat = mask.reshape(-1).gt(0.5)
    bit_count = int(flat.shape[0])
    packed = 0
    for idx, value in enumerate(flat.tolist()):
        if value:
            packed |= 1 << idx
    hex_width = max(1, (bit_count + 3) // 4)
    return f"{packed:0{hex_width}x}"


def _unpack_bool_mask_hex(bits: str, *, rows: int, cols: int) -> torch.Tensor:
    bit_count = int(rows * cols)
    packed = int(bits, 16) if bits else 0
    values = [(packed >> idx) & 1 for idx in range(bit_count)]
    return torch.tensor(values, dtype=torch.float32).view(rows, cols)


@lru_cache(maxsize=None)
def _cached_state_components(rows: int, cols: int) -> tuple[torch.Tensor, ...]:
    num_positions = rows * cols
    states = torch.arange(STATE_MODES * num_positions, dtype=torch.long)
    mode = states // num_positions
    pos = states % num_positions
    has_key = mode // 2
    reward_unlocked = mode % 2
    y = pos // cols
    x = pos % cols
    return states, has_key, reward_unlocked, y, x, pos


@lru_cache(maxsize=None)
def _cached_coordinate_bit_planes(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_positions = torch.arange(cols, dtype=torch.long).unsqueeze(0).expand(rows, cols)
    y_positions = torch.arange(rows, dtype=torch.long).unsqueeze(1).expand(rows, cols)
    x_bits = [
        torch.remainder(torch.div(x_positions, weight, rounding_mode="floor"), 2).to(torch.float32)
        for weight in COORD_BIT_WEIGHTS
    ]
    y_bits = [
        torch.remainder(torch.div(y_positions, weight, rounding_mode="floor"), 2).to(torch.float32)
        for weight in COORD_BIT_WEIGHTS
    ]
    return torch.stack(x_bits, dim=0), torch.stack(y_bits, dim=0)


def _sample_cell(
    rng: random.Random,
    rows: int,
    cols: int,
    *,
    exclude: set[tuple[int, int]] | None = None,
    region: str | None = None,
) -> tuple[int, int]:
    exclude = exclude or set()
    candidates: list[tuple[int, int]] = []
    for y in range(rows):
        for x in range(cols):
            if (y, x) in exclude:
                continue
            if region == "left" and x > cols // 3:
                continue
            if region == "right" and x < cols - cols // 3 - 1:
                continue
            if region == "top" and y > rows // 3:
                continue
            if region == "bottom" and y < rows - rows // 3 - 1:
                continue
            candidates.append((y, x))
    if not candidates:
        raise RuntimeError("No candidate cells available")
    return rng.choice(candidates)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _random_monotone_path(
    rng: random.Random,
    start: tuple[int, int],
    goal: tuple[int, int],
    rows: int,
    cols: int,
) -> list[tuple[int, int]]:
    y, x = start
    gy, gx = goal
    path = [(y, x)]
    budget = rows * cols * 4
    while (y, x) != (gy, gx) and budget > 0:
        budget -= 1
        options: list[tuple[int, int]] = []
        if y < gy:
            options.append((y + 1, x))
        if y > gy:
            options.append((y - 1, x))
        if x < gx:
            options.append((y, x + 1))
        if x > gx:
            options.append((y, x - 1))
        if len(options) >= 2 and rng.random() < 0.25:
            rng.shuffle(options)
        y, x = rng.choice(options)
        path.append((y, x))
    return path


def _bounded_random_simple_path(
    rng: random.Random,
    start: tuple[int, int],
    goal: tuple[int, int],
    rows: int,
    cols: int,
    *,
    max_path_length: int,
) -> list[tuple[int, int]]:
    """Sample a bounded, genuinely meandering simple path with randomized DFS.

    The returned path has length at most ``max_path_length`` and is guaranteed to
    be simple. We sample an exact target length first, then backtrack while
    pruning neighbors that cannot possibly reach the goal within the remaining
    budget. On these small grids this stays cheap and avoids the monotone-path
    artifact without rejection sampling over complete trap fields.
    """

    if start == goal:
        return [start]
    min_length = _manhattan(start, goal)
    if max_path_length <= min_length:
        return _random_monotone_path(rng, start, goal, rows, cols)

    candidate_lengths = [
        path_length
        for path_length in range(min_length, max_path_length + 1)
        if (path_length - min_length) % 2 == 0
    ]
    if not candidate_lengths:
        return _random_monotone_path(rng, start, goal, rows, cols)

    preferred_length = rng.choice(candidate_lengths)
    remaining_lengths = [length for length in candidate_lengths if length != preferred_length]
    rng.shuffle(remaining_lengths)
    trial_lengths = [preferred_length] + remaining_lengths

    def _dfs_exact_path(target_length: int) -> list[tuple[int, int]] | None:
        path = [start]
        visited = {start}

        def _search(current: tuple[int, int], steps_used: int) -> bool:
            steps_remaining = int(target_length) - int(steps_used)
            if current == goal:
                return steps_remaining == 0
            if steps_remaining <= 0:
                return False

            y, x = current
            current_dist = _manhattan(current, goal)
            neighbor_rows: list[tuple[float, tuple[int, int]]] = []
            for dy, dx in ACTION_DELTAS:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < rows and 0 <= nx < cols):
                    continue
                neighbor = (ny, nx)
                if neighbor in visited:
                    continue
                next_dist = _manhattan(neighbor, goal)
                remaining_after_move = steps_remaining - 1
                if next_dist > remaining_after_move:
                    continue
                if (remaining_after_move - next_dist) % 2 != 0:
                    continue
                # Randomized but mildly biased away from greedily collapsing to
                # the shortest path when there is still room to meander.
                dist_delta = next_dist - current_dist
                weight = 2.0 if dist_delta >= 0 else 1.0
                if neighbor == goal and remaining_after_move > 0:
                    weight *= 0.25
                neighbor_rows.append((weight, neighbor))

            if not neighbor_rows:
                return False

            remaining_rows = list(neighbor_rows)
            ordered_neighbors: list[tuple[int, int]] = []
            while remaining_rows:
                weights = [weight for weight, _ in remaining_rows]
                pick_index = rng.choices(range(len(remaining_rows)), weights=weights, k=1)[0]
                _weight, picked_neighbor = remaining_rows.pop(pick_index)
                ordered_neighbors.append(picked_neighbor)

            for neighbor in ordered_neighbors:
                visited.add(neighbor)
                path.append(neighbor)
                if _search(neighbor, steps_used + 1):
                    return True
                path.pop()
                visited.remove(neighbor)
            return False

        if _search(start, 0):
            return list(path)
        return None

    for target_length in trial_lengths:
        sampled = _dfs_exact_path(target_length)
        if sampled is not None:
            return sampled
    return _random_monotone_path(rng, start, goal, rows, cols)


def _topology_family(family_id: str) -> str:
    if family_id == "pressure_plate_open_room":
        return "open_room"
    if family_id == "pressure_plate_wall_detour":
        return "wall_detour"
    if family_id == "trap_open_room":
        return "open_room"
    return family_id


def _active_constraint_count_from_severity(severity: float) -> int:
    return max(0, min(6, int(round(float(severity) * 6.0))))


def _sample_unlock_bits(
    rng: random.Random,
    severity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    x_active = torch.zeros(3, dtype=torch.float32)
    x_value = torch.zeros(3, dtype=torch.float32)
    y_active = torch.zeros(3, dtype=torch.float32)
    y_value = torch.zeros(3, dtype=torch.float32)
    active_count = _active_constraint_count_from_severity(severity)
    bit_ids = [("x", bit_idx) for bit_idx in range(3)] + [("y", bit_idx) for bit_idx in range(3)]
    rng.shuffle(bit_ids)
    for axis, bit_idx in bit_ids[:active_count]:
        value = float(rng.randrange(0, 2))
        if axis == "x":
            x_active[bit_idx] = 1.0
            x_value[bit_idx] = value
        else:
            y_active[bit_idx] = 1.0
            y_value[bit_idx] = value
    return x_active, x_value, y_active, y_value, active_count


def _pressure_plate_mask_from_bits(
    rows: int,
    cols: int,
    x_active: torch.Tensor,
    x_value: torch.Tensor,
    y_active: torch.Tensor,
    y_value: torch.Tensor,
) -> torch.Tensor:
    mask = torch.ones(rows, cols, dtype=torch.float32)
    for y in range(rows):
        for x in range(cols):
            keep = 1.0
            for bit_idx, weight in enumerate(COORD_BIT_WEIGHTS):
                x_bit = float((x // weight) % 2)
                y_bit = float((y // weight) % 2)
                if float(x_active[bit_idx].item()) > 0.5 and x_bit != float(x_value[bit_idx].item()):
                    keep = 0.0
                    break
                if float(y_active[bit_idx].item()) > 0.5 and y_bit != float(y_value[bit_idx].item()):
                    keep = 0.0
                    break
            mask[y, x] = keep
    return mask


def _target_start_goal_distance(cfg: GridworldGeneratorConfig) -> int:
    max_grid_distance = cfg.rows + cfg.cols - 2
    max_horizon_distance = max(1, int(cfg.horizon) - 1)
    return max(1, min(int(cfg.min_start_goal_distance), max_grid_distance, max_horizon_distance))


def _base_backbone(
    cfg: GridworldGeneratorConfig,
    *,
    family_id: str,
    severity: float,
    backbone_seed: int,
) -> dict:
    rng = random.Random(backbone_seed)
    rows, cols = cfg.rows, cfg.cols
    target_distance = _target_start_goal_distance(cfg)
    topology_family = _topology_family(family_id)
    if topology_family in {"bridge_shortcut", "key_door", "slip_zone"}:
        orientation = "vertical" if rng.random() < 0.5 else "horizontal"
        if orientation == "vertical":
            start = _sample_cell(rng, rows, cols, region="left")
            goal = _sample_cell(rng, rows, cols, exclude={start}, region="right")
        else:
            start = _sample_cell(rng, rows, cols, region="top")
            goal = _sample_cell(rng, rows, cols, exclude={start}, region="bottom")
    elif topology_family == "decoy_reward":
        start = _sample_cell(rng, rows, cols, region="left")
        goal = _sample_cell(rng, rows, cols, exclude={start}, region="right")
    else:
        start = _sample_cell(rng, rows, cols)
        goal = _sample_cell(rng, rows, cols, exclude={start})
    while _manhattan(start, goal) < target_distance:
        goal = _sample_cell(rng, rows, cols, exclude={start})

    wall_mask = _empty_grid(rows, cols)
    path = _random_monotone_path(rng, start, goal, rows, cols)
    protected = set(path)
    protected.add(start)
    protected.add(goal)
    wall_density = rng.uniform(cfg.min_wall_density, cfg.max_wall_density)
    if topology_family in {"one_way_fork", "slip_zone"}:
        wall_density *= 0.5
    for y in range(rows):
        for x in range(cols):
            if (y, x) in protected:
                continue
            if rng.random() < wall_density:
                wall_mask[y, x] = 1.0
    return {
        "orientation": "vertical" if start[1] < goal[1] else "horizontal",
        "start": start,
        "goal": goal,
        "wall_mask": wall_mask,
    }


def _clear_cells(mask: torch.Tensor, cells: Iterable[tuple[int, int]]) -> None:
    for y, x in cells:
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            mask[y, x] = 0.0


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    y0, x0 = start
    y1, x1 = end
    out: list[tuple[int, int]] = []
    if y0 == y1:
        step = 1 if x1 >= x0 else -1
        for x in range(x0, x1 + step, step):
            out.append((y0, x))
    elif x0 == x1:
        step = 1 if y1 >= y0 else -1
        for y in range(y0, y1 + step, step):
            out.append((y, x0))
    return out


def _path_exists_without_hazards(
    rows: int,
    cols: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    wall_mask: torch.Tensor,
    hazard_mask: torch.Tensor,
) -> bool:
    queue = [start]
    seen = {start}
    while queue:
        y, x = queue.pop(0)
        if (y, x) == goal:
            return True
        for dy, dx in ACTION_DELTAS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < rows and 0 <= nx < cols):
                continue
            if (ny, nx) in seen:
                continue
            if float(wall_mask[ny, nx].item()) > 0.5:
                continue
            if float(hazard_mask[ny, nx].item()) > 0.5:
                continue
            seen.add((ny, nx))
            queue.append((ny, nx))
    return False


def _layout_from_backbone(
    cfg: GridworldGeneratorConfig,
    *,
    family_id: str,
    severity: float,
    backbone_seed: int,
    overlay_seed: int,
    slip_value: float = 0.0,
    wind_dir: int = 0,
    wind_strength: float = 0.0,
) -> GridworldLayout | None:
    if (slip_value != 0.0 or wind_strength != 0.0) and family_id != "trap_open_room":
        raise ValueError(
            f"Uniform slip/wind dynamics are only supported for trap_open_room, got family_id={family_id}"
        )
    if slip_value + wind_strength > 0.95:
        raise ValueError(
            f"slip_value + wind_strength must be <= 0.95, got {slip_value} + {wind_strength}"
        )
    if not 0 <= int(wind_dir) <= 3:
        raise ValueError(f"wind_dir must be in 0..3, got {wind_dir}")
    back = _base_backbone(cfg, family_id=family_id, severity=severity, backbone_seed=backbone_seed)
    rng = random.Random(overlay_seed)
    rows, cols = cfg.rows, cfg.cols
    target_distance = _target_start_goal_distance(cfg)
    wall_mask = back["wall_mask"].clone()
    goal_mask = _empty_grid(rows, cols)
    hazard_mask = _empty_grid(rows, cols)
    key_mask = _empty_grid(rows, cols)
    door_mask = _empty_grid(rows, cols)
    decoy_mask = _empty_grid(rows, cols)
    one_way_mask = torch.zeros(4, rows, cols, dtype=torch.float32)
    slip_prob = _empty_grid(rows, cols)
    start = back["start"]
    goal = back["goal"]
    goal_mask[goal] = 1.0
    subfamily_id = family_id
    catastrophe_penalty = cfg.catastrophe_penalty * (1.0 + 0.75 * severity)
    hazard_terminal = True
    decoy_reward = 0.0
    decoy_terminal = True
    goal_requires_unlock = False
    x_unlock_active = torch.zeros(3, dtype=torch.float32)
    x_unlock_value = torch.zeros(3, dtype=torch.float32)
    y_unlock_active = torch.zeros(3, dtype=torch.float32)
    y_unlock_value = torch.zeros(3, dtype=torch.float32)
    pressure_plate_mask = torch.ones(rows, cols, dtype=torch.float32)
    active_constraint_count = 0

    if family_id == "open_room":
        _clear_cells(wall_mask, [start, goal])
        ring = [(start[0] + dy, start[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        ring += [(goal[0] + dy, goal[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        _clear_cells(wall_mask, ring)
        subfamily_id = "open_room_sparse" if float(wall_mask.mean().item()) < 0.08 else "open_room_cluttered"

    elif family_id == "wall_detour":
        orientation = "vertical" if rng.random() < 0.5 else "horizontal"
        subfamily_id = f"wall_detour_{orientation}"
        if orientation == "vertical":
            x_barrier = min(max(1, (start[1] + goal[1]) // 2), cols - 2)
            gap_primary = rng.randrange(1, rows - 1)
            gap_secondary = None if severity > 0.45 else rng.randrange(1, rows - 1)
            for y in range(rows):
                if y == gap_primary or y == gap_secondary:
                    continue
                wall_mask[y, x_barrier] = 1.0
            _clear_cells(wall_mask, [start, goal, (gap_primary, x_barrier)])
        else:
            y_barrier = min(max(1, (start[0] + goal[0]) // 2), rows - 2)
            gap_primary = rng.randrange(1, cols - 1)
            gap_secondary = None if severity > 0.45 else rng.randrange(1, cols - 1)
            for x in range(cols):
                if x == gap_primary or x == gap_secondary:
                    continue
                wall_mask[y_barrier, x] = 1.0
            _clear_cells(wall_mask, [start, goal, (y_barrier, gap_primary)])

    elif family_id == "trap_open_room":
        wall_mask.zero_()
        protected_path = _bounded_random_simple_path(
            rng,
            start,
            goal,
            rows,
            cols,
            max_path_length=max(1, int(cfg.horizon) - 1),
        )
        protected_cells = set(protected_path)
        protected_cells.add(start)
        protected_cells.add(goal)
        trap_density = 0.05 + 0.70 * severity
        catastrophe_penalty = cfg.catastrophe_penalty
        hazard_terminal = False
        for y in range(rows):
            for x in range(cols):
                if (y, x) in protected_cells:
                    continue
                if rng.random() < trap_density:
                    hazard_mask[y, x] = 1.0
        subfamily_id = f"trap_open_room_d{int(round(100.0 * trap_density))}"
        if slip_value != 0.0:
            slip_prob.fill_(float(slip_value))
        if not _path_exists_without_hazards(rows, cols, start, goal, wall_mask, hazard_mask):
            return None

    elif family_id == "pressure_plate_open_room":
        _clear_cells(wall_mask, [start, goal])
        ring = [(start[0] + dy, start[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        ring += [(goal[0] + dy, goal[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        _clear_cells(wall_mask, ring)
        goal_requires_unlock = True
        x_unlock_active, x_unlock_value, y_unlock_active, y_unlock_value, active_constraint_count = _sample_unlock_bits(
            rng,
            severity,
        )
        pressure_plate_mask = _pressure_plate_mask_from_bits(
            rows,
            cols,
            x_unlock_active,
            x_unlock_value,
            y_unlock_active,
            y_unlock_value,
        )
        subfamily_id = f"pressure_plate_open_room_k{active_constraint_count}"

    elif family_id == "pressure_plate_wall_detour":
        orientation = "vertical" if rng.random() < 0.5 else "horizontal"
        if orientation == "vertical":
            x_barrier = min(max(1, (start[1] + goal[1]) // 2), cols - 2)
            gap_primary = rng.randrange(1, rows - 1)
            gap_secondary = None if severity > 0.35 else rng.randrange(1, rows - 1)
            for y in range(rows):
                if y == gap_primary or y == gap_secondary:
                    continue
                wall_mask[y, x_barrier] = 1.0
            _clear_cells(wall_mask, [start, goal, (gap_primary, x_barrier)])
        else:
            y_barrier = min(max(1, (start[0] + goal[0]) // 2), rows - 2)
            gap_primary = rng.randrange(1, cols - 1)
            gap_secondary = None if severity > 0.35 else rng.randrange(1, cols - 1)
            for x in range(cols):
                if x == gap_primary or x == gap_secondary:
                    continue
                wall_mask[y_barrier, x] = 1.0
            _clear_cells(wall_mask, [start, goal, (y_barrier, gap_primary)])
        goal_requires_unlock = True
        x_unlock_active, x_unlock_value, y_unlock_active, y_unlock_value, active_constraint_count = _sample_unlock_bits(
            rng,
            severity,
        )
        pressure_plate_mask = _pressure_plate_mask_from_bits(
            rows,
            cols,
            x_unlock_active,
            x_unlock_value,
            y_unlock_active,
            y_unlock_value,
        )
        subfamily_id = f"pressure_plate_wall_detour_{orientation}_k{active_constraint_count}"

    elif family_id == "bridge_shortcut":
        orientation = "vertical" if rng.random() < 0.5 else "horizontal"
        subfamily_id = f"bridge_shortcut_{orientation}"
        wall_mask.zero_()
        if orientation == "vertical":
            start = _sample_cell(rng, rows, cols, region="left")
            goal = _sample_cell(rng, rows, cols, exclude={start}, region="right")
            while _manhattan(start, goal) < target_distance:
                goal = _sample_cell(rng, rows, cols, exclude={start}, region="right")
            x_barrier = rng.randrange(cols // 2 - 1, cols // 2 + 1)
            strip_width = 1 if severity < 0.6 else 2
            bridge_row = rng.randrange(1, rows - 1)
            edge_gap_top = rng.random() < 0.5
            gap_rows = [0, 1] if edge_gap_top else [rows - 2, rows - 1]
            for dx in range(strip_width):
                xc = min(cols - 2, x_barrier + dx)
                for y in range(rows):
                    if y == bridge_row or y in gap_rows:
                        continue
                    hazard_mask[y, xc] = 1.0
        else:
            start = _sample_cell(rng, rows, cols, region="top")
            goal = _sample_cell(rng, rows, cols, exclude={start}, region="bottom")
            while _manhattan(start, goal) < target_distance:
                goal = _sample_cell(rng, rows, cols, exclude={start}, region="bottom")
            y_barrier = rng.randrange(rows // 2 - 1, rows // 2 + 1)
            strip_width = 1 if severity < 0.6 else 2
            bridge_col = rng.randrange(1, cols - 1)
            edge_gap_left = rng.random() < 0.5
            gap_cols = [0, 1] if edge_gap_left else [cols - 2, cols - 1]
            for dy in range(strip_width):
                yc = min(rows - 2, y_barrier + dy)
                for x in range(cols):
                    if x == bridge_col or x in gap_cols:
                        continue
                    hazard_mask[yc, x] = 1.0
        goal_mask.zero_()
        goal_mask[goal] = 1.0

    elif family_id == "key_door":
        orientation = "vertical" if rng.random() < 0.5 else "horizontal"
        subfamily_id = f"key_door_{orientation}"
        wall_mask.zero_()
        if orientation == "vertical":
            start = _sample_cell(rng, rows, cols, region="left")
            goal = _sample_cell(rng, rows, cols, exclude={start}, region="right")
            x_barrier = rng.randrange(cols // 2 - 1, cols // 2 + 1)
            door_row = rng.randrange(1, rows - 1)
            key_pos = _sample_cell(rng, rows, cols, exclude={start, goal, (door_row, x_barrier)}, region="left")
            for y in range(rows):
                if y == door_row:
                    continue
                wall_mask[y, x_barrier] = 1.0
            door_mask[door_row, x_barrier] = 1.0
            key_mask[key_pos] = 1.0
        else:
            start = _sample_cell(rng, rows, cols, region="top")
            goal = _sample_cell(rng, rows, cols, exclude={start}, region="bottom")
            y_barrier = rng.randrange(rows // 2 - 1, rows // 2 + 1)
            door_col = rng.randrange(1, cols - 1)
            key_pos = _sample_cell(rng, rows, cols, exclude={start, goal, (y_barrier, door_col)}, region="top")
            for x in range(cols):
                if x == door_col:
                    continue
                wall_mask[y_barrier, x] = 1.0
            door_mask[y_barrier, door_col] = 1.0
            key_mask[key_pos] = 1.0
        goal_mask.zero_()
        goal_mask[goal] = 1.0

    elif family_id == "one_way_fork":
        wall_mask.fill_(1.0)
        orientation = "vertical" if rng.random() < 0.5 else "horizontal"
        subfamily_id = f"one_way_fork_{orientation}"
        if orientation == "vertical":
            junction = (rng.randrange(2, rows - 2), rng.randrange(2, cols - 2))
            stem_len = rng.randrange(1, 3)
            branch_len_good = rng.randrange(2, 4)
            branch_len_bad = rng.randrange(1, 3)
            stem_cells = _line_cells((junction[0] + stem_len, junction[1]), junction)
            good_cells = _line_cells(junction, (junction[0], min(cols - 1, junction[1] + branch_len_good)))
            bad_cells = _line_cells(junction, (junction[0], max(0, junction[1] - branch_len_bad)))
            corridor = set(stem_cells + good_cells + bad_cells)
            _clear_cells(wall_mask, corridor)
            start = stem_cells[0]
            goal = good_cells[-1]
            hazard_mask[bad_cells[-1]] = 1.0
            for cell in good_cells[1:]:
                one_way_mask[3, cell[0], cell[1]] = 1.0
            for cell in bad_cells[1:]:
                one_way_mask[2, cell[0], cell[1]] = 1.0
        else:
            junction = (rng.randrange(2, rows - 2), rng.randrange(2, cols - 2))
            stem_len = rng.randrange(1, 3)
            branch_len_good = rng.randrange(2, 4)
            branch_len_bad = rng.randrange(1, 3)
            stem_cells = _line_cells((junction[0], junction[1] + stem_len), junction)
            good_cells = _line_cells(junction, (max(0, junction[0] - branch_len_good), junction[1]))
            bad_cells = _line_cells(junction, (min(rows - 1, junction[0] + branch_len_bad), junction[1]))
            corridor = set(stem_cells + good_cells + bad_cells)
            _clear_cells(wall_mask, corridor)
            start = stem_cells[0]
            goal = good_cells[-1]
            hazard_mask[bad_cells[-1]] = 1.0
            for cell in good_cells[1:]:
                one_way_mask[0, cell[0], cell[1]] = 1.0
            for cell in bad_cells[1:]:
                one_way_mask[1, cell[0], cell[1]] = 1.0
        goal_mask.zero_()
        goal_mask[goal] = 1.0

    elif family_id == "slip_zone":
        orientation = "vertical" if rng.random() < 0.5 else "horizontal"
        subfamily_id = f"slip_zone_{orientation}"
        wall_mask.zero_()
        slip_value = 0.10 + 0.55 * severity
        if orientation == "vertical":
            start = _sample_cell(rng, rows, cols, region="left")
            goal = _sample_cell(rng, rows, cols, exclude={start}, region="right")
            center_x = rng.randrange(cols // 2 - 1, cols // 2 + 1)
            band_y0 = rng.randrange(1, rows - 3)
            band_y1 = min(rows - 1, band_y0 + 2 + int(severity > 0.6))
            for y in range(band_y0, band_y1):
                for x in range(max(1, center_x - 1), min(cols - 1, center_x + 2)):
                    slip_prob[y, x] = slip_value
                    if x == min(cols - 2, center_x + 1):
                        hazard_mask[y, x] = 1.0
        else:
            start = _sample_cell(rng, rows, cols, region="top")
            goal = _sample_cell(rng, rows, cols, exclude={start}, region="bottom")
            center_y = rng.randrange(rows // 2 - 1, rows // 2 + 1)
            band_x0 = rng.randrange(1, cols - 3)
            band_x1 = min(cols - 1, band_x0 + 2 + int(severity > 0.6))
            for y in range(max(1, center_y - 1), min(rows - 1, center_y + 2)):
                for x in range(band_x0, band_x1):
                    slip_prob[y, x] = slip_value
                    if y == min(rows - 2, center_y + 1):
                        hazard_mask[y, x] = 1.0
        goal_mask.zero_()
        goal_mask[goal] = 1.0

    elif family_id == "decoy_reward":
        subfamily_id = "decoy_reward_terminal" if severity > 0.55 else "decoy_reward_nonterminal"
        decoy_terminal = severity > 0.55
        decoy_reward = 0.25 + 0.55 * severity
        wall_mask = wall_mask * 0.5
        _clear_cells(wall_mask, [start, goal])
        exclude = {start, goal}
        decoy = _sample_cell(rng, rows, cols, exclude=exclude)
        while _manhattan(start, decoy) > max(2, cfg.rows // 3):
            decoy = _sample_cell(rng, rows, cols, exclude=exclude)
        decoy_mask[decoy] = 1.0

    else:
        raise ValueError(f"Unknown family: {family_id}")

    layout = GridworldLayout(
        family_id=family_id,
        subfamily_id=subfamily_id,
        severity=severity,
        backbone_seed=backbone_seed,
        overlay_seed=overlay_seed,
        rows=rows,
        cols=cols,
        horizon=cfg.horizon,
        gamma=cfg.gamma,
        step_cost=cfg.step_cost,
        goal_reward=cfg.goal_reward,
        catastrophe_penalty=catastrophe_penalty,
        hazard_terminal=hazard_terminal,
        decoy_reward=decoy_reward,
        decoy_terminal=decoy_terminal,
        goal_requires_unlock=goal_requires_unlock,
        x_unlock_active=x_unlock_active.clone(),
        x_unlock_value=x_unlock_value.clone(),
        y_unlock_active=y_unlock_active.clone(),
        y_unlock_value=y_unlock_value.clone(),
        pressure_plate_mask=pressure_plate_mask.clamp(0.0, 1.0),
        active_constraint_count=int(active_constraint_count),
        start_pos=start,
        wall_mask=wall_mask.clamp(0.0, 1.0),
        goal_mask=goal_mask.clamp(0.0, 1.0),
        hazard_mask=hazard_mask.clamp(0.0, 1.0),
        key_mask=key_mask.clamp(0.0, 1.0),
        door_mask=door_mask.clamp(0.0, 1.0),
        decoy_mask=decoy_mask.clamp(0.0, 1.0),
        one_way_mask=one_way_mask.clamp(0.0, 1.0),
        slip_prob=slip_prob.clamp(0.0, 0.95),
        wind_dir=int(wind_dir),
        wind_strength=float(wind_strength),
    )
    return layout if layout_is_valid(layout) else None


def _cell_is_pressure_plate_layout(layout: GridworldLayout, y: int, x: int) -> bool:
    return float(layout.pressure_plate_mask[y, x].item()) > 0.5


def _cell_is_pressure_plate_task(task: GridworldTask, y: int, x: int) -> bool:
    return float(task.pressure_plate_mask[y, x].item()) > 0.5


def _initial_reward_unlocked_layout(layout: GridworldLayout) -> int:
    return int((not layout.goal_requires_unlock) or layout.active_constraint_count == 0)


def _initial_reward_unlocked_task(task: GridworldTask) -> int:
    return int((not task.goal_requires_unlock) or task.active_constraint_count == 0)


def _goal_active_layout(layout: GridworldLayout, reward_unlocked: int) -> bool:
    return bool(reward_unlocked or (not layout.goal_requires_unlock))


def _goal_active_task(task: GridworldTask, reward_unlocked: int) -> bool:
    return bool(reward_unlocked or (not task.goal_requires_unlock))


def _terminal_cell(layout: GridworldLayout, reward_unlocked: int, y: int, x: int) -> bool:
    if float(layout.goal_mask[y, x].item()) > 0.5 and _goal_active_layout(layout, reward_unlocked):
        return True
    if layout.hazard_terminal and float(layout.hazard_mask[y, x].item()) > 0.5:
        return True
    if layout.decoy_terminal and float(layout.decoy_mask[y, x].item()) > 0.5:
        return True
    return False


def _terminal_task_cell(task: GridworldTask, reward_unlocked: int, y: int, x: int) -> bool:
    if float(task.goal_mask[y, x].item()) > 0.5 and _goal_active_task(task, reward_unlocked):
        return True
    if task.hazard_terminal and float(task.hazard_mask[y, x].item()) > 0.5:
        return True
    if task.decoy_terminal and float(task.decoy_mask[y, x].item()) > 0.5:
        return True
    return False


def _passable(layout: GridworldLayout, has_key: int, y: int, x: int) -> bool:
    if not (0 <= y < layout.rows and 0 <= x < layout.cols):
        return False
    if float(layout.wall_mask[y, x].item()) > 0.5:
        return False
    if float(layout.door_mask[y, x].item()) > 0.5 and not has_key:
        return False
    if layout.hazard_terminal and float(layout.hazard_mask[y, x].item()) > 0.5:
        return False
    if layout.decoy_terminal and float(layout.decoy_mask[y, x].item()) > 0.5:
        return False
    return True


def layout_is_valid(layout: GridworldLayout) -> bool:
    sy, sx = layout.start_pos
    if not _passable(layout, 0, sy, sx):
        return False
    if float(layout.goal_mask.sum().item()) != 1.0:
        return False
    if float(layout.goal_mask[sy, sx].item()) > 0.5:
        return False
    if layout.goal_requires_unlock and layout.active_constraint_count > 0 and _cell_is_pressure_plate_layout(layout, *torch.nonzero(layout.goal_mask, as_tuple=False)[0].tolist()):
        return False
    start_state = (0, _initial_reward_unlocked_layout(layout), sy, sx, 0)
    queue = [start_state]
    seen = {start_state}
    goal_reached = False
    while queue:
        has_key, reward_unlocked, y, x, steps_used = queue.pop(0)
        if float(layout.goal_mask[y, x].item()) > 0.5 and _goal_active_layout(layout, reward_unlocked):
            goal_reached = True
            break
        if steps_used >= int(layout.horizon):
            continue
        allowed = [0, 1, 2, 3]
        if float(layout.one_way_mask[:, y, x].sum().item()) > 0.5:
            allowed = [i for i in range(4) if float(layout.one_way_mask[i, y, x].item()) > 0.5]
        for action in allowed:
            dy, dx = ACTION_DELTAS[action]
            ny, nx = y + dy, x + dx
            if not _passable(layout, has_key, ny, nx):
                continue
            next_has_key = has_key or int(float(layout.key_mask[ny, nx].item()) > 0.5)
            next_reward_unlocked = reward_unlocked or int(_cell_is_pressure_plate_layout(layout, ny, nx))
            nxt = (next_has_key, next_reward_unlocked, ny, nx, steps_used + 1)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return goal_reached


def canonical_layout_hash(layout: GridworldLayout) -> str:
    hasher = hashlib.sha256()
    # Deduplicate by realized MDP semantics rather than generator provenance.
    # If two sampling paths produce the same environment dynamics/rewards, they
    # should collide here even when severity or seeds differ.
    meta = {
        "rows": int(layout.rows),
        "cols": int(layout.cols),
        "horizon": int(layout.horizon),
        "gamma": float(layout.gamma),
        "step_cost": float(layout.step_cost),
        "goal_reward": float(layout.goal_reward),
        "catastrophe_penalty": float(layout.catastrophe_penalty),
        "hazard_terminal": bool(layout.hazard_terminal),
        "decoy_reward": float(layout.decoy_reward),
        "decoy_terminal": bool(layout.decoy_terminal),
        "goal_requires_unlock": bool(layout.goal_requires_unlock),
        "start_pos": [int(layout.start_pos[0]), int(layout.start_pos[1])],
    }
    # Wind enters the hash only when active so that every pre-wind layout
    # (legacy and slip banks) keeps its published hash exactly.
    if float(layout.wind_strength) != 0.0:
        meta["wind_dir"] = int(layout.wind_dir)
        meta["wind_strength"] = float(layout.wind_strength)
    hasher.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
    for tensor in (
        layout.wall_mask,
        layout.goal_mask,
        layout.hazard_mask,
        layout.key_mask,
        layout.door_mask,
        layout.decoy_mask,
        layout.one_way_mask,
        layout.slip_prob,
        layout.x_unlock_active,
        layout.x_unlock_value,
        layout.y_unlock_active,
        layout.y_unlock_value,
        layout.pressure_plate_mask,
    ):
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()[:24]


def layout_feature_tensor(layout: GridworldLayout, *, blind_dynamics: bool = False) -> torch.Tensor:
    """Spatial observation for the policy.

    blind_dynamics=True zeroes the slip channel (wind is never observed), so
    the policy sees layout geometry only while the transition spec keeps the
    true dynamics -- the information-withholding mode where per-manifest
    regret has an irreducible hedging floor.
    """
    x_coord_bits, y_coord_bits = _cached_coordinate_bit_planes(layout.rows, layout.cols)
    slip_channel = (
        torch.zeros_like(layout.slip_prob) if blind_dynamics else layout.slip_prob
    )
    return torch.cat(
        [
            layout.wall_mask.unsqueeze(0),
            layout.goal_mask.unsqueeze(0),
            layout.hazard_mask.unsqueeze(0),
            layout.key_mask.unsqueeze(0),
            layout.door_mask.unsqueeze(0),
            layout.decoy_mask.unsqueeze(0),
            layout.one_way_mask,
            slip_channel.unsqueeze(0),
            x_coord_bits,
            y_coord_bits,
        ],
        dim=0,
    )


def layout_task_vector(layout: GridworldLayout, *, include_slip: bool = False) -> torch.Tensor:
    family_one_hot = torch.zeros(len(POLICY_FAMILIES), dtype=torch.float32)
    policy_family_id = POLICY_FAMILY_ALIAS.get(layout.family_id, layout.family_id)
    family_one_hot[POLICY_FAMILY_TO_INDEX[policy_family_id]] = 1.0
    scalars = torch.tensor(
        [
            float(layout.goal_requires_unlock),
            float(layout.decoy_terminal),
            float(layout.severity),
            float(layout.step_cost),
            float(layout.goal_reward),
            float(layout.catastrophe_penalty),
            float(layout.decoy_reward),
            float(layout.active_constraint_count) / 6.0,
        ],
        dtype=torch.float32,
    )
    parts = [
        scalars,
        layout.x_unlock_active.to(torch.float32),
        layout.x_unlock_value.to(torch.float32),
        layout.y_unlock_active.to(torch.float32),
        layout.y_unlock_value.to(torch.float32),
        family_one_hot,
    ]
    if include_slip:
        parts.append(torch.tensor([float(layout.slip_prob.mean().item())], dtype=torch.float32))
    return torch.cat(parts, dim=0)


def _deterministic_transition(
    layout: GridworldLayout,
    *,
    has_key: int,
    reward_unlocked: int,
    y: int,
    x: int,
    action: int,
) -> tuple[int, int, int, int, float]:
    if _terminal_cell(layout, reward_unlocked, y, x):
        return has_key, reward_unlocked, y, x, 0.0
    if float(layout.one_way_mask[:, y, x].sum().item()) > 0.5 and float(layout.one_way_mask[action, y, x].item()) <= 0.5:
        ny, nx = y, x
    else:
        dy, dx = ACTION_DELTAS[action]
        ny = min(max(y + dy, 0), layout.rows - 1)
        nx = min(max(x + dx, 0), layout.cols - 1)
        if float(layout.wall_mask[ny, nx].item()) > 0.5:
            ny, nx = y, x
        elif float(layout.door_mask[ny, nx].item()) > 0.5 and not has_key:
            ny, nx = y, x
    next_has_key = has_key or int(float(layout.key_mask[ny, nx].item()) > 0.5)
    next_reward_unlocked = reward_unlocked or int(_cell_is_pressure_plate_layout(layout, ny, nx))
    reward = layout.step_cost
    if float(layout.decoy_mask[ny, nx].item()) > 0.5:
        reward += layout.decoy_reward
    if float(layout.goal_mask[ny, nx].item()) > 0.5 and _goal_active_layout(layout, next_reward_unlocked):
        reward += layout.goal_reward
    if float(layout.hazard_mask[ny, nx].item()) > 0.5:
        reward += layout.catastrophe_penalty
    return next_has_key, next_reward_unlocked, ny, nx, reward


def build_dense_transition_spec(layout: GridworldLayout) -> tuple[torch.Tensor, torch.Tensor, int]:
    rows, cols = layout.rows, layout.cols
    num_states = STATE_MODES * rows * cols
    probs = torch.zeros(num_states, 4, num_states, dtype=torch.float32)
    rewards = torch.zeros(num_states, 4, num_states, dtype=torch.float32)
    wall = layout.wall_mask.gt(0.5).tolist()
    goal = layout.goal_mask.gt(0.5).tolist()
    hazard = layout.hazard_mask.gt(0.5).tolist()
    key = layout.key_mask.gt(0.5).tolist()
    door = layout.door_mask.gt(0.5).tolist()
    decoy = layout.decoy_mask.gt(0.5).tolist()
    one_way = layout.one_way_mask.gt(0.5).tolist()
    slip = layout.slip_prob.tolist()
    plate = layout.pressure_plate_mask.gt(0.5).tolist()

    def is_terminal_pos(reward_unlocked: int, y: int, x: int) -> bool:
        return bool(
            (goal[y][x] and _goal_active_layout(layout, reward_unlocked))
            or (layout.hazard_terminal and hazard[y][x])
            or (layout.decoy_terminal and decoy[y][x])
        )

    for state in range(num_states):
        has_key, reward_unlocked, y, x = decode_state(state, rows, cols)
        if is_terminal_pos(reward_unlocked, y, x):
            probs[state, :, state] = 1.0
            continue
        slip_value = float(slip[y][x])
        wind_strength = float(layout.wind_strength)
        for intended_action in range(4):
            action_weights = [slip_value / 4.0] * 4
            action_weights[intended_action] += 1.0 - slip_value - wind_strength
            if wind_strength != 0.0:
                action_weights[int(layout.wind_dir)] += wind_strength
            for actual_action, weight in enumerate(action_weights):
                if any(one_way[a][y][x] for a in range(4)) and not one_way[actual_action][y][x]:
                    ny, nx = y, x
                else:
                    dy, dx = ACTION_DELTAS[actual_action]
                    ny = min(max(y + dy, 0), rows - 1)
                    nx = min(max(x + dx, 0), cols - 1)
                    if wall[ny][nx] or (door[ny][nx] and not has_key):
                        ny, nx = y, x
                next_has_key = has_key or int(key[ny][nx])
                next_reward_unlocked = reward_unlocked or int(plate[ny][nx])
                reward = float(layout.step_cost)
                if decoy[ny][nx]:
                    reward += float(layout.decoy_reward)
                if goal[ny][nx] and _goal_active_layout(layout, next_reward_unlocked):
                    reward += float(layout.goal_reward)
                if hazard[ny][nx]:
                    reward += float(layout.catastrophe_penalty)
                next_state = state_index(next_has_key, next_reward_unlocked, ny, nx, rows, cols)
                probs[state, intended_action, next_state] += float(weight)
                rewards[state, intended_action, next_state] = float(reward)
    start_state = state_index(
        0,
        _initial_reward_unlocked_layout(layout),
        layout.start_pos[0],
        layout.start_pos[1],
        rows,
        cols,
    )
    return probs, rewards, start_state


def compute_optimal_value(
    transition_next_states: torch.Tensor,
    transition_probs: torch.Tensor,
    transition_rewards: torch.Tensor,
    *,
    start_state_index: int,
    horizon: int,
    gamma: float,
) -> float:
    num_states = transition_probs.shape[0]
    values = torch.zeros(num_states, dtype=torch.float32)
    for _ in range(horizon):
        next_values = values[transition_next_states]
        q = (transition_probs * (transition_rewards + float(gamma) * next_values)).sum(dim=-1)
        values = q.max(dim=-1).values
    return float(values[start_state_index].item())


def build_transition_spec(layout: GridworldLayout) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    rows, cols = layout.rows, layout.cols
    num_positions = rows * cols
    states, has_key, reward_unlocked, y, x, pos = _cached_state_components(rows, cols)
    num_states = int(states.shape[0])
    device = layout.wall_mask.device
    next_states = torch.zeros(num_states, 4, MAX_TRANSITION_BRANCHING, dtype=torch.long, device=device)
    probs = torch.zeros(num_states, 4, MAX_TRANSITION_BRANCHING, dtype=torch.float32, device=device)
    rewards = torch.zeros(num_states, 4, MAX_TRANSITION_BRANCHING, dtype=torch.float32, device=device)

    wall = layout.wall_mask.reshape(-1).gt(0.5)
    goal = layout.goal_mask.reshape(-1).gt(0.5)
    hazard = layout.hazard_mask.reshape(-1).gt(0.5)
    key = layout.key_mask.reshape(-1).gt(0.5)
    door = layout.door_mask.reshape(-1).gt(0.5)
    decoy = layout.decoy_mask.reshape(-1).gt(0.5)
    plate = layout.pressure_plate_mask.reshape(-1).gt(0.5)
    one_way = layout.one_way_mask.reshape(4, -1).gt(0.5)
    slip = layout.slip_prob.reshape(-1).to(torch.float32)

    has_key_bool = has_key.to(torch.bool)
    reward_unlocked_bool = reward_unlocked.to(torch.bool)
    pos_goal = goal[pos]
    pos_hazard = hazard[pos]
    pos_decoy = decoy[pos]
    goal_active_now = reward_unlocked_bool | (not layout.goal_requires_unlock)
    terminal_mask = pos_goal & goal_active_now
    if layout.hazard_terminal:
        terminal_mask = terminal_mask | pos_hazard
    if layout.decoy_terminal:
        terminal_mask = terminal_mask | pos_decoy

    deltas = torch.tensor(ACTION_DELTAS, dtype=torch.long, device=device)
    y_long = y.to(device=device)
    x_long = x.to(device=device)
    pos_long = pos.to(device=device)
    has_key_device = has_key_bool.to(device=device)
    reward_unlocked_device = reward_unlocked_bool.to(device=device)

    next_y = (y_long.unsqueeze(0) + deltas[:, 0].unsqueeze(1)).clamp_(0, rows - 1)
    next_x = (x_long.unsqueeze(0) + deltas[:, 1].unsqueeze(1)).clamp_(0, cols - 1)
    candidate_pos = next_y * cols + next_x

    current_one_way = one_way[:, pos_long]
    blocked_by_one_way = current_one_way.any(dim=0, keepdim=True) & (~current_one_way)
    blocked_by_cell = wall[candidate_pos] | (door[candidate_pos] & (~has_key_device.unsqueeze(0)))
    blocked = blocked_by_one_way | blocked_by_cell
    final_pos = torch.where(blocked, pos_long.unsqueeze(0), candidate_pos)

    next_has_key = has_key_device.unsqueeze(0) | key[final_pos]
    next_reward_unlocked = reward_unlocked_device.unsqueeze(0) | plate[final_pos]
    goal_active_next = next_reward_unlocked | (not layout.goal_requires_unlock)

    reward_values = torch.full((4, num_states), float(layout.step_cost), dtype=torch.float32, device=device)
    if float(layout.decoy_reward) != 0.0:
        reward_values = reward_values + decoy[final_pos].to(torch.float32) * float(layout.decoy_reward)
    reward_values = reward_values + (goal[final_pos] & goal_active_next).to(torch.float32) * float(layout.goal_reward)
    reward_values = reward_values + hazard[final_pos].to(torch.float32) * float(layout.catastrophe_penalty)

    next_state_values = (
        (next_has_key.to(torch.long) * 2 + next_reward_unlocked.to(torch.long)) * num_positions
        + final_pos.to(torch.long)
    )
    slip_by_state = slip[pos_long]
    wind_strength = float(layout.wind_strength)
    action_weights = slip_by_state.unsqueeze(0).unsqueeze(0).expand(4, 4, num_states).clone() / 4.0
    diagonal_boost = (1.0 - slip_by_state - wind_strength).to(torch.float32)
    for intended_action in range(4):
        action_weights[intended_action, intended_action] += diagonal_boost
    if wind_strength != 0.0:
        action_weights[:, int(layout.wind_dir)] += wind_strength

    sorted_next_states, permutation = torch.sort(next_state_values, dim=0)
    sorted_rewards = reward_values.gather(0, permutation)
    is_new_group = torch.ones_like(sorted_next_states, dtype=torch.bool)
    is_new_group[1:] = sorted_next_states[1:] != sorted_next_states[:-1]
    duplicate_mask = ~is_new_group
    if duplicate_mask.any():
        duplicate_rewards_match = torch.isclose(
            sorted_rewards[1:],
            sorted_rewards[:-1],
            rtol=0.0,
            atol=1e-7,
        )
        if not bool(duplicate_rewards_match[duplicate_mask[1:]].all().item()):
            raise RuntimeError("Non-coalescible transition rewards for identical successor state")
    group_index = is_new_group.to(torch.long).cumsum(dim=0) - 1
    branching_needed = int(group_index.max().item()) + 1
    if branching_needed > MAX_TRANSITION_BRANCHING:
        raise RuntimeError(
            f"Transition branching {branching_needed} exceeds fixed budget {MAX_TRANSITION_BRANCHING}"
        )
    flat_state_group_index = torch.arange(num_states, device=device).unsqueeze(0)
    for intended_action in range(4):
        branch_probs = action_weights[intended_action]
        sorted_probs = branch_probs.gather(0, permutation)
        scatter_index = (flat_state_group_index * MAX_TRANSITION_BRANCHING + group_index).reshape(-1)
        probs_flat = torch.zeros(num_states * MAX_TRANSITION_BRANCHING, dtype=torch.float32, device=device)
        rewards_flat = torch.zeros_like(probs_flat)
        next_states_flat = torch.zeros(num_states * MAX_TRANSITION_BRANCHING, dtype=torch.long, device=device)
        probs_flat.scatter_add_(0, scatter_index, sorted_probs.reshape(-1))
        rewards_flat.scatter_(0, scatter_index, sorted_rewards.reshape(-1))
        next_states_flat.scatter_(0, scatter_index, sorted_next_states.reshape(-1))
        probs[:, intended_action] = probs_flat.view(num_states, MAX_TRANSITION_BRANCHING)
        rewards[:, intended_action] = rewards_flat.view(num_states, MAX_TRANSITION_BRANCHING)
        next_states[:, intended_action] = next_states_flat.view(num_states, MAX_TRANSITION_BRANCHING)

    terminal_mask_device = terminal_mask.to(device=device)
    if terminal_mask_device.any():
        next_states[terminal_mask_device] = 0
        probs[terminal_mask_device] = 0.0
        rewards[terminal_mask_device] = 0.0
        next_states[terminal_mask_device, :, 0] = states.to(device=device)[terminal_mask_device].unsqueeze(1)
        probs[terminal_mask_device, :, 0] = 1.0
    start_state = state_index(
        0,
        _initial_reward_unlocked_layout(layout),
        layout.start_pos[0],
        layout.start_pos[1],
        rows,
        cols,
    )
    return next_states, probs, rewards, start_state


def reconstruct_dense_transition_spec(
    transition_next_states: torch.Tensor,
    transition_probs: torch.Tensor,
    transition_rewards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_states = transition_probs.shape[0]
    dense_probs = torch.zeros(num_states, 4, num_states, dtype=torch.float32)
    dense_rewards = torch.zeros(num_states, 4, num_states, dtype=torch.float32)
    for state in range(num_states):
        for action in range(4):
            for branch in range(transition_probs.shape[-1]):
                prob = float(transition_probs[state, action, branch].item())
                if prob <= 0.0:
                    continue
                next_state = int(transition_next_states[state, action, branch].item())
                dense_probs[state, action, next_state] += prob
                dense_rewards[state, action, next_state] = float(
                    transition_rewards[state, action, branch].item()
                )
    return dense_probs, dense_rewards


def build_task_spec_from_layout(layout: GridworldLayout) -> GridworldTaskSpec:
    transition_next_states, transition_probs, transition_rewards, start_state = build_transition_spec(layout)
    canonical_hash = canonical_layout_hash(layout)
    optimal_value = compute_optimal_value(
        transition_next_states,
        transition_probs,
        transition_rewards,
        start_state_index=start_state,
        horizon=layout.horizon,
        gamma=layout.gamma,
    )
    return GridworldTaskSpec(
        family_id=layout.family_id,
        subfamily_id=layout.subfamily_id,
        severity=float(layout.severity),
        backbone_seed=int(layout.backbone_seed),
        overlay_seed=int(layout.overlay_seed),
        canonical_hash=canonical_hash,
        optimal_value=float(optimal_value),
    )


def build_trap_layout_record_from_layout(
    layout: GridworldLayout, *, dynamics_id: int = -1
) -> TrapLayoutRecord:
    if layout.family_id != "trap_open_room":
        raise ValueError(f"Trap layout records only support trap_open_room, got {layout.family_id}")
    transition_next_states, transition_probs, transition_rewards, start_state = build_transition_spec(layout)
    canonical_hash = canonical_layout_hash(layout)
    optimal_value = compute_optimal_value(
        transition_next_states,
        transition_probs,
        transition_rewards,
        start_state_index=start_state,
        horizon=layout.horizon,
        gamma=layout.gamma,
    )
    goal_positions = torch.nonzero(layout.goal_mask > 0.5, as_tuple=False)
    if goal_positions.shape[0] != 1:
        raise ValueError("Trap layout record expects exactly one goal cell")
    goal_pos = (int(goal_positions[0, 0].item()), int(goal_positions[0, 1].item()))
    slip_unique = torch.unique(layout.slip_prob)
    if slip_unique.numel() != 1:
        raise ValueError(
            "Trap layout records only support a uniform slip_prob grid; "
            f"found {slip_unique.numel()} distinct values"
        )
    return TrapLayoutRecord(
        family_id=layout.family_id,
        subfamily_id=layout.subfamily_id,
        severity=float(layout.severity),
        canonical_hash=canonical_hash,
        optimal_value=float(optimal_value),
        start_pos=(int(layout.start_pos[0]), int(layout.start_pos[1])),
        goal_pos=goal_pos,
        hazard_bits=_pack_bool_mask_hex(layout.hazard_mask),
        backbone_seed=int(layout.backbone_seed),
        overlay_seed=int(layout.overlay_seed),
        slip_value=float(slip_unique.item()),
        wind_dir=int(layout.wind_dir),
        wind_strength=float(layout.wind_strength),
        dynamics_id=int(dynamics_id),
    )


def build_task_from_layout(
    layout: GridworldLayout,
    *,
    split_name: str,
    task_id: str,
    optimal_value_override: float | None = None,
    canonical_hash_override: str | None = None,
    include_slip_in_task_vector: bool = False,
    blind_dynamics: bool = False,
) -> GridworldTask:
    if blind_dynamics and include_slip_in_task_vector:
        raise ValueError("blind_dynamics and include_slip_in_task_vector are contradictory")
    transition_next_states, transition_probs, transition_rewards, start_state = build_transition_spec(layout)
    feature_tensor = layout_feature_tensor(layout, blind_dynamics=blind_dynamics)
    task_vector = layout_task_vector(layout, include_slip=include_slip_in_task_vector)
    canonical_hash = canonical_hash_override or canonical_layout_hash(layout)
    optimal_value = (
        float(optimal_value_override)
        if optimal_value_override is not None
        else compute_optimal_value(
            transition_next_states,
            transition_probs,
            transition_rewards,
            start_state_index=start_state,
            horizon=layout.horizon,
            gamma=layout.gamma,
        )
    )
    return GridworldTask(
        task_id=task_id,
        split_name=split_name,
        family_id=layout.family_id,
        subfamily_id=layout.subfamily_id,
        severity=float(layout.severity),
        backbone_seed=int(layout.backbone_seed),
        overlay_seed=int(layout.overlay_seed),
        canonical_hash=canonical_hash,
        rows=layout.rows,
        cols=layout.cols,
        horizon=layout.horizon,
        gamma=float(layout.gamma),
        step_cost=float(layout.step_cost),
        goal_reward=float(layout.goal_reward),
        catastrophe_penalty=float(layout.catastrophe_penalty),
        hazard_terminal=bool(layout.hazard_terminal),
        decoy_reward=float(layout.decoy_reward),
        decoy_terminal=bool(layout.decoy_terminal),
        goal_requires_unlock=bool(layout.goal_requires_unlock),
        x_unlock_active=layout.x_unlock_active.clone(),
        x_unlock_value=layout.x_unlock_value.clone(),
        y_unlock_active=layout.y_unlock_active.clone(),
        y_unlock_value=layout.y_unlock_value.clone(),
        pressure_plate_mask=layout.pressure_plate_mask.clone(),
        active_constraint_count=int(layout.active_constraint_count),
        start_pos=layout.start_pos,
        wall_mask=layout.wall_mask.clone(),
        goal_mask=layout.goal_mask.clone(),
        hazard_mask=layout.hazard_mask.clone(),
        key_mask=layout.key_mask.clone(),
        door_mask=layout.door_mask.clone(),
        decoy_mask=layout.decoy_mask.clone(),
        one_way_mask=layout.one_way_mask.clone(),
        slip_prob=layout.slip_prob.clone(),
        feature_tensor=feature_tensor,
        task_vector=task_vector,
        transition_next_states=transition_next_states,
        transition_probs=transition_probs,
        transition_rewards=transition_rewards,
        start_state_index=start_state,
        optimal_value=float(optimal_value),
        wind_dir=int(layout.wind_dir),
        wind_strength=float(layout.wind_strength),
    )


def materialize_task_from_spec(
    spec: GridworldTaskSpec,
    *,
    cfg: GridworldGeneratorConfig,
    split_name: str,
    task_id: str,
    include_slip_in_task_vector: bool = False,
    blind_dynamics: bool = False,
) -> GridworldTask:
    layout = generate_layout(
        spec.family_id,
        float(spec.severity),
        cfg=cfg,
        backbone_seed=int(spec.backbone_seed),
        overlay_seed=int(spec.overlay_seed),
    )
    if layout is None:
        raise RuntimeError(
            f"Unable to regenerate task from spec for family={spec.family_id} "
            f"backbone_seed={spec.backbone_seed} overlay_seed={spec.overlay_seed}"
        )
    canonical_hash = canonical_layout_hash(layout)
    if canonical_hash != spec.canonical_hash:
        raise RuntimeError(
            "Regenerated task hash mismatch: "
            f"expected {spec.canonical_hash}, got {canonical_hash}"
        )
    return build_task_from_layout(
        layout,
        split_name=split_name,
        task_id=task_id,
        optimal_value_override=float(spec.optimal_value),
        canonical_hash_override=canonical_hash,
        include_slip_in_task_vector=include_slip_in_task_vector,
        blind_dynamics=blind_dynamics,
    )


def _layout_from_trap_layout_record(
    record: TrapLayoutRecord,
    *,
    cfg: GridworldGeneratorConfig,
) -> GridworldLayout:
    rows, cols = cfg.rows, cfg.cols
    goal_mask = _empty_grid(rows, cols)
    goal_mask[record.goal_pos] = 1.0
    return GridworldLayout(
        family_id=record.family_id,
        subfamily_id=record.subfamily_id,
        severity=float(record.severity),
        backbone_seed=int(record.backbone_seed),
        overlay_seed=int(record.overlay_seed),
        rows=rows,
        cols=cols,
        horizon=cfg.horizon,
        gamma=cfg.gamma,
        step_cost=cfg.step_cost,
        goal_reward=cfg.goal_reward,
        catastrophe_penalty=cfg.catastrophe_penalty,
        hazard_terminal=False,
        decoy_reward=0.0,
        decoy_terminal=True,
        goal_requires_unlock=False,
        x_unlock_active=torch.zeros(3, dtype=torch.float32),
        x_unlock_value=torch.zeros(3, dtype=torch.float32),
        y_unlock_active=torch.zeros(3, dtype=torch.float32),
        y_unlock_value=torch.zeros(3, dtype=torch.float32),
        pressure_plate_mask=torch.ones(rows, cols, dtype=torch.float32),
        active_constraint_count=0,
        start_pos=(int(record.start_pos[0]), int(record.start_pos[1])),
        wall_mask=_empty_grid(rows, cols),
        goal_mask=goal_mask,
        hazard_mask=_unpack_bool_mask_hex(record.hazard_bits, rows=rows, cols=cols),
        key_mask=_empty_grid(rows, cols),
        door_mask=_empty_grid(rows, cols),
        decoy_mask=_empty_grid(rows, cols),
        one_way_mask=torch.zeros(4, rows, cols, dtype=torch.float32),
        slip_prob=torch.full((rows, cols), float(record.slip_value), dtype=torch.float32),
        wind_dir=int(record.wind_dir),
        wind_strength=float(record.wind_strength),
    )


def materialize_task_from_trap_layout_record(
    record: TrapLayoutRecord,
    *,
    cfg: GridworldGeneratorConfig,
    split_name: str,
    task_id: str,
    include_slip_in_task_vector: bool = False,
    blind_dynamics: bool = False,
) -> GridworldTask:
    layout = _layout_from_trap_layout_record(record, cfg=cfg)
    return build_task_from_layout(
        layout,
        split_name=split_name,
        task_id=task_id,
        optimal_value_override=float(record.optimal_value),
        canonical_hash_override=record.canonical_hash,
        include_slip_in_task_vector=include_slip_in_task_vector,
        blind_dynamics=blind_dynamics,
    )


def generate_layout(
    family_id: str,
    severity: float,
    *,
    cfg: GridworldGeneratorConfig,
    backbone_seed: int,
    overlay_seed: int,
    slip_value: float = 0.0,
    wind_dir: int = 0,
    wind_strength: float = 0.0,
) -> GridworldLayout | None:
    severity = min(max(float(severity), 0.0), 1.0)
    return _layout_from_backbone(
        cfg,
        family_id=family_id,
        severity=severity,
        backbone_seed=backbone_seed,
        overlay_seed=overlay_seed,
        slip_value=slip_value,
        wind_dir=wind_dir,
        wind_strength=wind_strength,
    )


def sample_unique_tasks(
    *,
    split_name: str,
    slices: Sequence[FamilySlice],
    cfg: GridworldGeneratorConfig,
    seed: int,
    globally_seen_hashes: set[str] | None = None,
) -> list[GridworldTask]:
    rng = random.Random(seed)
    tasks: list[GridworldTask] = []
    seen = globally_seen_hashes if globally_seen_hashes is not None else set()
    for slice_idx, family_slice in enumerate(slices):
        print(
            f"    [{split_name}] family={family_slice.family_id} count={family_slice.count}",
            flush=True,
        )
        for task_idx in range(family_slice.count):
            accepted = False
            for _attempt in range(cfg.max_generation_attempts_per_task):
                severity = rng.uniform(family_slice.severity_min, family_slice.severity_max)
                backbone_seed = rng.randrange(0, 2**31 - 1)
                overlay_seed = rng.randrange(0, 2**31 - 1)
                layout = generate_layout(
                    family_slice.family_id,
                    severity,
                    cfg=cfg,
                    backbone_seed=backbone_seed,
                    overlay_seed=overlay_seed,
                )
                if layout is None:
                    continue
                canonical_hash = canonical_layout_hash(layout)
                if canonical_hash in seen:
                    continue
                task_id = f"{split_name}:{family_slice.family_id}:{slice_idx}:{task_idx}:{canonical_hash}"
                task = build_task_from_layout(layout, split_name=split_name, task_id=task_id)
                tasks.append(task)
                seen.add(canonical_hash)
                accepted = True
                break
            if not accepted:
                raise RuntimeError(
                    f"Unable to sample a valid unique task for split={split_name} "
                    f"family={family_slice.family_id} after {cfg.max_generation_attempts_per_task} attempts"
                )
        print(
            f"    [{split_name}] family={family_slice.family_id} ready",
            flush=True,
        )
    return tasks


def sample_unique_task_specs(
    *,
    split_name: str,
    slices: Sequence[FamilySlice],
    cfg: GridworldGeneratorConfig,
    seed: int,
    globally_seen_hashes: set[str] | None = None,
) -> list[GridworldTaskSpec]:
    rng = random.Random(seed)
    specs: list[GridworldTaskSpec] = []
    seen = globally_seen_hashes if globally_seen_hashes is not None else set()
    for slice_idx, family_slice in enumerate(slices):
        print(
            f"    [{split_name}] family={family_slice.family_id} count={family_slice.count}",
            flush=True,
        )
        for task_idx in range(family_slice.count):
            accepted = False
            for _attempt in range(cfg.max_generation_attempts_per_task):
                severity = rng.uniform(family_slice.severity_min, family_slice.severity_max)
                backbone_seed = rng.randrange(0, 2**31 - 1)
                overlay_seed = rng.randrange(0, 2**31 - 1)
                layout = generate_layout(
                    family_slice.family_id,
                    severity,
                    cfg=cfg,
                    backbone_seed=backbone_seed,
                    overlay_seed=overlay_seed,
                )
                if layout is None:
                    continue
                canonical_hash = canonical_layout_hash(layout)
                if canonical_hash in seen:
                    continue
                spec = build_task_spec_from_layout(layout)
                specs.append(spec)
                seen.add(canonical_hash)
                accepted = True
                break
            if not accepted:
                raise RuntimeError(
                    f"Unable to sample a valid unique task for split={split_name} "
                    f"family={family_slice.family_id} after {cfg.max_generation_attempts_per_task} attempts"
                )
        print(
            f"    [{split_name}] family={family_slice.family_id} ready",
            flush=True,
        )
    return specs


def sample_unique_trap_layout_records(
    *,
    split_name: str,
    slices: Sequence[FamilySlice],
    cfg: GridworldGeneratorConfig,
    seed: int,
    globally_seen_hashes: set[str] | None = None,
) -> list[TrapLayoutRecord]:
    rng = random.Random(seed)
    records: list[TrapLayoutRecord] = []
    seen = globally_seen_hashes if globally_seen_hashes is not None else set()
    for slice_idx, family_slice in enumerate(slices):
        if family_slice.family_id != "trap_open_room":
            raise ValueError(
                f"Trap layout records only support trap_open_room, got {family_slice.family_id}"
            )
        print(
            f"    [{split_name}] family={family_slice.family_id} count={family_slice.count}",
            flush=True,
        )
        for task_idx in range(family_slice.count):
            accepted = False
            for _attempt in range(cfg.max_generation_attempts_per_task):
                severity = rng.uniform(family_slice.severity_min, family_slice.severity_max)
                backbone_seed = rng.randrange(0, 2**31 - 1)
                overlay_seed = rng.randrange(0, 2**31 - 1)
                layout = generate_layout(
                    family_slice.family_id,
                    severity,
                    cfg=cfg,
                    backbone_seed=backbone_seed,
                    overlay_seed=overlay_seed,
                    slip_value=family_slice.slip_value,
                    wind_dir=family_slice.wind_dir,
                    wind_strength=family_slice.wind_strength,
                )
                if layout is None:
                    continue
                canonical_hash = canonical_layout_hash(layout)
                if canonical_hash in seen:
                    continue
                record = build_trap_layout_record_from_layout(
                    layout, dynamics_id=family_slice.dynamics_id
                )
                records.append(record)
                seen.add(canonical_hash)
                accepted = True
                break
            if not accepted:
                raise RuntimeError(
                    f"Unable to sample a valid unique task for split={split_name} "
                    f"family={family_slice.family_id} after {cfg.max_generation_attempts_per_task} attempts"
                )
        print(
            f"    [{split_name}] family={family_slice.family_id} ready",
            flush=True,
        )
    return records


def prepare_task_batch(tasks: Sequence[GridworldTask], device: str) -> dict[str, torch.Tensor]:
    features = torch.stack([task.feature_tensor for task in tasks], dim=0).to(device)
    task_vectors = torch.stack([task.task_vector for task in tasks], dim=0).to(device)
    next_states = torch.stack([task.transition_next_states for task in tasks], dim=0).to(device)
    transitions = torch.stack([task.transition_probs for task in tasks], dim=0).to(device)
    rewards = torch.stack([task.transition_rewards for task in tasks], dim=0).to(device)
    starts = torch.tensor([task.start_state_index for task in tasks], dtype=torch.long, device=device)
    optimal_values = torch.tensor([task.optimal_value for task in tasks], dtype=torch.float32, device=device)
    return {
        "features": features,
        "task_vectors": task_vectors,
        "transition_next_states": next_states,
        "transition_probs": transitions,
        "transition_rewards": rewards,
        "start_state_indices": starts,
        "optimal_values": optimal_values,
    }


def _gather_sparse_next_values(
    values: torch.Tensor,
    transition_next_states: torch.Tensor,
) -> torch.Tensor:
    batch_size = values.shape[0]
    gathered = values.gather(1, transition_next_states.view(batch_size, -1))
    return gathered.view_as(transition_next_states)


def evaluate_dense_policy_exact(
    dense_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    horizon: int,
    gamma: float,
) -> torch.Tensor:
    state_logits = dense_logits_to_state_action_logits(
        dense_logits,
        rows=dense_logits.shape[-2],
        cols=dense_logits.shape[-1],
    )
    transition_next_states = batch["transition_next_states"]
    transition_probs = batch["transition_probs"]
    transition_rewards = batch["transition_rewards"]
    values = torch.zeros(
        transition_probs.shape[0],
        transition_probs.shape[1],
        dtype=torch.float32,
        device=transition_probs.device,
    )
    for t in range(1, horizon + 1):
        next_values = _gather_sparse_next_values(values, transition_next_states)
        q = (transition_probs * (transition_rewards + float(gamma) * next_values)).sum(dim=-1)
        pi = F.softmax(state_logits[:, t], dim=-1)
        values = (pi * q).sum(dim=-1)
    starts = batch["start_state_indices"].unsqueeze(1)
    return values.gather(1, starts).squeeze(1)


def optimal_values_for_batch(batch: dict[str, torch.Tensor], *, horizon: int, gamma: float) -> torch.Tensor:
    transition_next_states = batch["transition_next_states"]
    transition_probs = batch["transition_probs"]
    transition_rewards = batch["transition_rewards"]
    values = torch.zeros(
        transition_probs.shape[0],
        transition_probs.shape[1],
        dtype=torch.float32,
        device=transition_probs.device,
    )
    for _ in range(horizon):
        next_values = _gather_sparse_next_values(values, transition_next_states)
        q = (transition_probs * (transition_rewards + float(gamma) * next_values)).sum(dim=-1)
        values = q.max(dim=-1).values
    starts = batch["start_state_indices"].unsqueeze(1)
    return values.gather(1, starts).squeeze(1)


def sample_policy_rollouts(
    tasks: Sequence[GridworldTask],
    dense_logits: torch.Tensor,
    *,
    n_rollouts: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_logits = dense_logits_to_state_action_logits(
        dense_logits,
        rows=dense_logits.shape[-2],
        cols=dense_logits.shape[-1],
    )
    returns: list[torch.Tensor] = []
    logprob_sums: list[torch.Tensor] = []
    for task_index, task in enumerate(tasks):
        next_states = task.transition_next_states.to(device)
        probs = task.transition_probs.to(device)
        rewards = task.transition_rewards.to(device)
        task_returns: list[torch.Tensor] = []
        task_logprobs: list[torch.Tensor] = []
        for _ in range(n_rollouts):
            state = int(task.start_state_index)
            total_return = torch.zeros((), dtype=torch.float32, device=device)
            total_logprob = torch.zeros((), dtype=torch.float32, device=device)
            for t in range(task.horizon, 0, -1):
                logits = state_logits[task_index, t, state]
                action_dist = torch.distributions.Categorical(logits=logits)
                action = action_dist.sample()
                total_logprob = total_logprob + action_dist.log_prob(action)
                branch_dist = torch.distributions.Categorical(probs=probs[state, action])
                branch = branch_dist.sample()
                next_state = next_states[state, action, branch]
                total_return = total_return + rewards[state, action, branch]
                state = int(next_state.item())
                _has_key, reward_unlocked, y, x = decode_state(state, task.rows, task.cols)
                if _terminal_task_cell(task, reward_unlocked, y, x):
                    break
            task_returns.append(total_return)
            task_logprobs.append(total_logprob)
        returns.append(torch.stack(task_returns, dim=0))
        logprob_sums.append(torch.stack(task_logprobs, dim=0))
    return torch.stack(returns, dim=0), torch.stack(logprob_sums, dim=0)


def terminal_state_mask(tasks: Sequence[GridworldTask], device: str) -> torch.Tensor:
    """Vectorized `_terminal_task_cell` over the dense flat state space.

    Returns a bool tensor of shape [num_tasks, num_states] in the same
    state-index layout as `state_index` (mode-major, then row-major cells).
    """
    goal = torch.stack([task.goal_mask.reshape(-1) for task in tasks], dim=0).to(device) > 0.5
    hazard = torch.stack([task.hazard_mask.reshape(-1) for task in tasks], dim=0).to(device) > 0.5
    decoy = torch.stack([task.decoy_mask.reshape(-1) for task in tasks], dim=0).to(device) > 0.5
    hazard_terminal = torch.tensor(
        [bool(task.hazard_terminal) for task in tasks], dtype=torch.bool, device=device
    ).unsqueeze(1)
    decoy_terminal = torch.tensor(
        [bool(task.decoy_terminal) for task in tasks], dtype=torch.bool, device=device
    ).unsqueeze(1)
    goal_requires_unlock = torch.tensor(
        [bool(task.goal_requires_unlock) for task in tasks], dtype=torch.bool, device=device
    ).unsqueeze(1)
    cell_terminal = (hazard & hazard_terminal) | (decoy & decoy_terminal)
    per_mode: list[torch.Tensor] = []
    for mode in range(STATE_MODES):
        reward_unlocked = bool(mode % 2)
        if reward_unlocked:
            goal_active = goal
        else:
            goal_active = goal & ~goal_requires_unlock
        per_mode.append(goal_active | cell_terminal)
    return torch.cat(per_mode, dim=1)


def sample_policy_rollouts_batched(
    tasks: Sequence[GridworldTask],
    dense_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    n_rollouts: int,
    device: str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched drop-in for `sample_policy_rollouts` (same [T, R] contract).

    Vectorizes over tasks x rollouts and loops only over the horizon.
    Distributionally identical to the loop sampler (not RNG-stream
    identical), with one generalization: rewards are discounted by
    gamma**step, which coincides with the loop sampler at gamma=1.0
    (the only value used in practice; the loop sampler ignores gamma).
    Gradient flows through the accumulated action log-probs only.
    """
    state_logits = dense_logits_to_state_action_logits(
        dense_logits,
        rows=dense_logits.shape[-2],
        cols=dense_logits.shape[-1],
    )
    num_tasks = len(tasks)
    horizon = int(tasks[0].horizon)
    gamma = float(tasks[0].gamma)
    next_states = batch["transition_next_states"]
    probs = batch["transition_probs"]
    rewards = batch["transition_rewards"]
    term_mask = terminal_state_mask(tasks, device)
    task_idx = torch.arange(num_tasks, dtype=torch.long, device=device).repeat_interleave(n_rollouts)
    state = batch["start_state_indices"].repeat_interleave(n_rollouts)
    alive = torch.ones(num_tasks * n_rollouts, dtype=torch.bool, device=device)
    total_return = torch.zeros(num_tasks * n_rollouts, dtype=torch.float32, device=device)
    total_logprob = torch.zeros(num_tasks * n_rollouts, dtype=torch.float32, device=device)
    for step, t in enumerate(range(horizon, 0, -1)):
        cur_logits = state_logits[task_idx, t, state]
        logp = F.log_softmax(cur_logits, dim=-1)
        with torch.no_grad():
            action = torch.multinomial(logp.detach().exp(), 1, generator=generator).squeeze(1)
        chosen_logp = logp.gather(1, action.unsqueeze(1)).squeeze(1)
        branch_probs = probs[task_idx, state, action]
        branch = torch.multinomial(branch_probs, 1, generator=generator).squeeze(1)
        next_state = next_states[task_idx, state, action].gather(1, branch.unsqueeze(1)).squeeze(1)
        reward = rewards[task_idx, state, action].gather(1, branch.unsqueeze(1)).squeeze(1)
        alive_f = alive.to(torch.float32)
        total_return = total_return + (gamma**step) * reward * alive_f
        total_logprob = total_logprob + chosen_logp * alive_f
        state = torch.where(alive, next_state, state)
        alive = alive & ~term_mask[task_idx, state]
        if not bool(alive.any()):
            break
    return (
        total_return.view(num_tasks, n_rollouts),
        total_logprob.view(num_tasks, n_rollouts),
    )


def run_family_support_audit(
    *,
    cfg: GridworldGeneratorConfig,
    seed: int,
    normal_sample_count: int,
    challenge_sample_count: int,
    planned_usage_by_family: dict[str, int],
    families: Sequence[str] | None = None,
    normal_min_unique_per_family: int = 50000,
    rare_min_unique_base: int = 5000,
) -> dict:
    rng = random.Random(seed)
    results: dict[str, dict[str, float]] = {}
    normal_unique_hashes: set[str] = set()
    overall_unique_hashes: set[str] = set()
    family_ids = list(ALL_FAMILIES if families is None else families)
    for family_id in family_ids:
        sample_count = normal_sample_count if family_id in NORMAL_FAMILIES else challenge_sample_count
        seen_hashes: set[str] = set()
        accepted = 0
        collisions = 0
        for _ in range(sample_count):
            severity = rng.random()
            backbone_seed = rng.randrange(0, 2**31 - 1)
            overlay_seed = rng.randrange(0, 2**31 - 1)
            layout = generate_layout(
                family_id,
                severity,
                cfg=cfg,
                backbone_seed=backbone_seed,
                overlay_seed=overlay_seed,
            )
            if layout is None:
                continue
            accepted += 1
            h = canonical_layout_hash(layout)
            if h in seen_hashes:
                collisions += 1
            else:
                seen_hashes.add(h)
        unique_count = len(seen_hashes)
        acceptance_rate = accepted / max(1, sample_count)
        collision_rate = collisions / max(1, accepted)
        results[family_id] = {
            "sample_count": float(sample_count),
            "accepted_count": float(accepted),
            "unique_count": float(unique_count),
            "acceptance_rate": float(acceptance_rate),
            "collision_rate": float(collision_rate),
            "min_required_unique": float(
                int(normal_min_unique_per_family)
                if family_id in NORMAL_FAMILIES
                else max(int(rare_min_unique_base), 20 * int(planned_usage_by_family.get(family_id, 0)))
            ),
        }
        overall_unique_hashes.update(seen_hashes)
        if family_id in NORMAL_FAMILIES:
            normal_unique_hashes.update(seen_hashes)
    return {
        "family_results": results,
        "overall_unique_total": float(len(overall_unique_hashes)),
        "normal_unique_total": float(len(normal_unique_hashes)),
    }
