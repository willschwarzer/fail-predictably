"""Search for an empirical feasible benchmark for the minimum expected MSE.

This is numerical search, not a proof of a minimum. Independent evaluation
estimates the expected MSE of a feasible vector, hence an upper bound on the
minimum; finite evaluation draws do not themselves give a certified bound.

Examples (run computation inside an existing CPU allocation):
  python tail_opt.py 'lqpw_pw2apsp_seed17:100' --pool password_slot_26 --label smoke
  python tail_opt.py --fleet --bank --prepare-only --label fleet
  python tail_opt.py --manifest tail_opt_results/fleet.manifest.npz --shard 0/125
  python tail_opt.py --manifest tail_opt_results/fleet.manifest.npz --aggregate-only
  python tail_opt.py --fleet --bank --band fitmax --prepare-only --label fleet_fitmax
  python tail_opt.py --grid --grid-pools 8 --prepare-only
  python tail_opt.py --manifest tail_opt_results/grid.manifest.npz --shard 0/168
  python tail_opt.py --manifest tail_opt_results/grid.manifest.npz --aggregate-only

Grid mode streams optimization/evaluation permutations, caps optimization draws
at max(256, 2000000 // k), and preserves all independent evaluation draws. It
checks base and certificate scoring MSE against cert_scale.csv in either band.

Each manifest freezes steps, arrays, configuration and code identity. Shards
share it and write independent atomic pool records. Failed records are retried;
completed records are validated and skipped. Aggregation never scores a failed
pool. Use a different label for a new configuration or changed inputs/code.

--self-test runs small numerical and record-integrity checks. --interrupt-after
terminates the process immediately after a pool record is committed, for a real
interruption/resume test. Compare resumed and uninterrupted CSVs with the same
configuration and --pool selectors; CSVs omit timings and storage paths.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import tempfile
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
from scipy.optimize import isotonic_regression
import torch

HERE = Path(__file__).resolve().parent
# Resolved, so code_identity's relative paths work when REPO_ROOT is a symlinked submit directory.
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3])).resolve()
for _path in (REPO_ROOT / "src", HERE):
    sys.path.insert(0, str(_path))

from cert_scale import MARGIN, certify, pools_from_bank, smoke
from cert_sq_strategy import involved_sigma
from feasibility import project
from forecast_experiment.dynamics_probe import involved_sigma_reference
from forecast_experiment.ols_utils import count_extrapolated_eval_points
from heldout_spread import pools
from poolscale_readout import split_errors
import tail_opt_grid as grid

RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts/results"))
RUNS = RESULTS_ROOT / "lquant_dynamics/runs"
READOUTS = RESULTS_ROOT / "lquant_dynamics/readouts"
SCHEMA = 1
RANKS = np.array([1, 2, 5, 10, 20, 50, 100, 200])
MODES = ("optimization", "evaluation", "legacy")
NOTICE = ("Empirical feasible benchmark: independent evaluation estimates an upper "
          "bound on the minimum expected MSE; numerical search is not a proof of a minimum.")
CONFIG_NAMES = ("band", "opt_splits", "eval_splits", "opt_seed", "eval_seed", "search_seed",
                "iterations", "random_starts", "learning_rate", "slope_points", "refine_points",
                "pool_seconds", "threads")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_arrays(meta, arrays):
    h = hashlib.sha256(canonical(meta).encode())
    for name, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        h.update(canonical([name, a.dtype.str, list(a.shape)]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def code_identity():
    paths = [Path(__file__).resolve(), HERE / "tail_opt_grid.py", HERE / "cert_scale.py", HERE / "feasibility.py",
             HERE / "cert_sq_strategy.py",
             HERE / "heldout_spread.py",
             HERE / "g_transfer_k.py",
             HERE / "g_transfer_v2.py",
             REPO_ROOT / "src/forecast_experiment/dynamics_probe.py",
             REPO_ROOT / "src/forecast_experiment/ols_utils.py",
             HERE / "poolscale_readout.py"]
    return {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_record(path, meta, arrays):
    meta = dict(meta, payload_digest=digest_arrays(meta, arrays))
    buf = io.BytesIO()
    np.savez_compressed(buf, metadata=np.array(canonical(meta)), **arrays)
    atomic_bytes(path, buf.getvalue())


def load_record(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        arrays = {name: z[name] for name in z.files if name != "metadata"}
    saved = meta.pop("payload_digest")
    if digest_arrays(meta, arrays) != saved or meta["schema"] != SCHEMA:
        raise ValueError(f"record checksum/schema mismatch: {path}")
    return meta, arrays


def permutations(n, count, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n) for _ in range(count)])


class Gather:
    """Selected rank ids for sorted values, or value-ordered prompt ids for legacy checks."""

    def __init__(self, perms, M, k, e=1):
        self.n, self.M, self.k = perms.shape[1], M, k
        self.fr = np.sort(perms[:, :M], axis=1)[:, :k].copy()
        self.dr = np.sort(perms[:, M:], axis=1)[:, :e].copy()
        self.y = np.log(np.arange(1, k + 1) / (M + 1.0))
        self.ym = self.y.mean()
        self.target = math.log(1.0 / (self.n - M))
        self.tf, self.td = torch.from_numpy(self.fr), torch.from_numpy(self.dr)
        self.ty = torch.from_numpy(self.y - self.ym)

    @classmethod
    def streamed(cls, n, count, seed, M, k, e=1):
        """Keep selected ranks only, with exactly the full-permutation RNG sequence."""
        rng = np.random.default_rng(seed)
        fr, dr = np.empty((count, k), dtype=np.int64), np.empty((count, e), dtype=np.int64)
        for i in range(count):
            p = rng.permutation(n)
            fr[i] = np.sort(np.partition(p[:M], k - 1)[:k])
            dr[i] = np.sort(np.partition(p[M:], e - 1)[:e])
        obj = cls(np.arange(n)[None, :], M, k, e)
        obj.fr, obj.dr = fr, dr
        obj.tf, obj.td = torch.from_numpy(fr), torch.from_numpy(dr)
        return obj

    @classmethod
    def assigned(cls, v, perms, M, k, e=1):
        order = np.argsort(-v, kind="stable")
        rank_of = np.empty(v.size, dtype=np.int64)
        rank_of[order] = np.arange(v.size)
        obj = cls(rank_of[perms], M, k, e)
        obj.fr, obj.dr = order[obj.fr], order[obj.dr]
        obj.tf, obj.td = torch.from_numpy(obj.fr), torch.from_numpy(obj.dr)
        return obj

    def errors(self, v):
        x = -np.log(np.maximum(-v[self.fr], 1e-30))
        xm = x.mean(axis=1, keepdims=True)
        sxx = ((x - xm) ** 2).sum(axis=1)
        sxy = ((x - xm) * (self.y - self.ym)).sum(axis=1)
        ok = sxx > 1e-20
        b = np.where(ok, sxy / np.where(ok, sxx, 1.0), 0.0)
        a = self.ym - b * xm[:, 0]
        nz = b != 0.0
        xstar = np.where(nz, (self.target - a) / np.where(nz, b, 1.0), x[:, 0])
        with np.errstate(over="ignore", invalid="ignore"):
            return (-np.exp(-xstar) - v[self.dr[:, 0]]) / math.log(10.0)

    def torch_errors(self, v):
        x = -torch.log(torch.clamp(-v[self.tf], min=1e-30))
        xm = x.mean(dim=1, keepdim=True)
        sxx = ((x - xm) ** 2).sum(dim=1)
        sxy = ((x - xm) * self.ty).sum(dim=1)
        ok = sxx > 1e-20
        b = torch.where(ok, sxy / torch.where(ok, sxx, 1.0), 0.0)
        a = self.ym - b * xm[:, 0]
        nz = b != 0.0
        xstar = torch.where(nz, (self.target - a) / torch.where(nz, b, 1.0), x[:, 0])
        return (-torch.exp(-xstar) - v[self.td[:, 0]]) / math.log(10.0)

    def parts(self, v):
        f, d = v[self.fr], v[self.dr]
        return {"fit": float(f.std(axis=1).mean()), "deploy": float(d.std(axis=1).mean()),
                "fitmax": float(np.concatenate([f, d[:, :1]], axis=1).std(axis=1).mean()),
                "involved": float(np.concatenate([f, d], axis=1).std(axis=1).mean())}

    def sigma(self, v, band):
        d = self.dr if band == "involved" else self.dr[:, :1]
        return float(np.concatenate([v[self.fr], v[d]], axis=1).std(axis=1).mean())

    def torch_sigma(self, v, band):
        d = self.td if band == "involved" else self.td[:, :1]
        return torch.cat([v[self.tf], v[d]], dim=1).std(dim=1, correction=0).mean()


def feasible(v, base, band, kind, lo, hi):
    if (not np.isfinite(v).all() or np.any(v >= 0) or np.any(np.diff(v) > 0)
            or np.any(v > base + 1e-9)):
        return False
    s = band.sigma(v, kind)
    return math.isfinite(s) and lo * (1 - 1e-9) <= s <= hi * (1 + 1e-9)


def exact_scale(u, base, band, kind, lo, hi):
    if not np.isfinite(u).all() or np.any(u >= 0) or np.any(np.diff(u) > 0):
        return None
    s = band.sigma(u, kind)
    if not math.isfinite(s) or (s == 0 and lo > 0):
        return None
    c = max(lo / s if s else 0.0, float(np.max(base / u)))
    if not math.isfinite(c) or c * s > hi * (1 + 1e-12):
        return None
    v = c * u
    return v if feasible(v, base, band, kind, lo, hi) else None


def torch_scaled(z, base, band, kind, lo):
    u = -torch.exp(z)
    s = band.torch_sigma(u, kind)
    floor_scale = lo / s if lo > 0 else s * 0
    c = torch.maximum(floor_scale, torch.max(base / u))
    return c * u


def mse(errors):
    if not np.isfinite(errors).all():
        raise ValueError("non-finite forecast")
    out = float(np.mean(errors ** 2))
    if not math.isfinite(out):
        raise ValueError("non-finite MSE")
    return out


def forecast_check(label, v, perms, M, k):
    g = Gather.assigned(v, perms, M, k)
    expected = split_errors(v, perms, M, k=k)[0]
    actual = g.torch_errors(torch.from_numpy(np.ascontiguousarray(v))).detach().numpy()
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise AssertionError(f"non-finite forecast in {label}")
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, err_msg=label)
    np.testing.assert_allclose(g.errors(v), expected, rtol=1e-10, atol=1e-10, err_msg=label)
    return {"case": label, "max_abs": float(np.max(np.abs(actual - expected)))}


def corner_checks():
    n, M, k = 31, 12, 5
    p = permutations(n, 12, 917)
    cases = {
        "ties": -np.ones(n) * 3,
        "plateau": -np.r_[np.ones(20), np.linspace(2, 5, 11)],
        "nearly_tied_fallback": -np.exp(np.arange(n) * 1e-13),
        "nearly_tied_sxx_branch": -np.exp(np.arange(n) * 1e-8),
        "log_floor": -np.geomspace(1e-35, 1e-28, n),
    }
    checks = [forecast_check(name, v, p, M, k) for name, v in cases.items()]
    tied_fit = -np.r_[np.full(M, 3.0), 1.0, np.full(n - M - 1, 4.0)]
    tied_perms = np.arange(n)[None, :]
    tied_gather = Gather.assigned(tied_fit, tied_perms, M, k, 3)
    assert tied_gather.parts(tied_fit)["involved"] > 0
    checks.append(forecast_check("tied_fit_positive_band", tied_fit, tied_perms, M, k))
    # These smooth inputs keep fit membership and both forecast branches fixed.
    v = -np.exp(np.linspace(-0.7, 2.0, n))
    g = Gather(p, M, k, 3)
    t = torch.tensor(v, requires_grad=True)
    loss = g.torch_errors(t).square().mean()
    loss.backward()
    rng = np.random.default_rng(281)
    for j in range(3):
        direction = rng.normal(size=n)
        direction /= np.linalg.norm(direction)
        h = 1e-5
        fd = (mse(g.errors(v + h * direction)) - mse(g.errors(v - h * direction))) / (2 * h)
        ad = float(t.grad.numpy() @ direction)
        np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)
        checks.append({"case": f"forecast_gradient_{j}", "max_abs": abs(ad - fd)})
    # Check differentiation through the active lower-band scale as well.
    z = np.log(-v)
    base = v * 0.2
    lo = 2 * g.sigma(v, "involved")
    t = torch.tensor(z, requires_grad=True)
    out = torch_scaled(t, torch.from_numpy(base), g, "involved", lo)
    g.torch_errors(out).square().mean().backward()
    for j in range(3):
        direction = rng.normal(size=n)
        direction /= np.linalg.norm(direction)
        h = 1e-5
        values = [exact_scale(-np.exp(z + sign * h * direction), base, g, "involved", lo, lo * 1.1)
                  for sign in (1, -1)]
        fd = (mse(g.errors(values[0])) - mse(g.errors(values[1]))) / (2 * h)
        ad = float(t.grad.numpy() @ direction)
        np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)
        checks.append({"case": f"scaled_gradient_{j}", "max_abs": abs(ad - fd)})
    base = -(-v) ** 1.2  # The last rank uniquely determines the no-raise scale.
    lo = 0.1 * g.sigma(v, "involved")
    t = torch.tensor(z, requires_grad=True)
    out = torch_scaled(t, torch.from_numpy(base), g, "involved", lo)
    g.torch_errors(out).square().mean().backward()
    direction = rng.normal(size=n)
    direction /= np.linalg.norm(direction)
    h = 1e-5
    values = [exact_scale(-np.exp(z + sign * h * direction), base, g, "involved", lo, 100.0)
              for sign in (1, -1)]
    fd = (mse(g.errors(values[0])) - mse(g.errors(values[1]))) / (2 * h)
    ad = float(t.grad.numpy() @ direction)
    np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)
    checks.append({"case": "no_raise_scale_gradient", "max_abs": abs(ad - fd)})
    return checks


def band_project(v, base, band, kind, lo, hi):
    """The feasibility.py sequence, with its band functional replaced for fitmax."""
    sig = band.sigma(np.sort(v)[::-1], kind)
    if not math.isfinite(sig) or sig <= 0:
        raise ValueError("trained band clamp has zero or non-finite sigma")
    v = v * (min(max(sig, lo), hi) / sig)
    v = np.minimum(v, base)
    for _ in range(50):
        sig = band.sigma(np.sort(v)[::-1], kind)
        if sig < lo * (1 - 1e-9):
            if sig <= 0:
                break
            v = v * (lo / sig)
        elif sig > hi * (1 + 1e-9):
            v = np.minimum(v * (hi / sig), base)
        else:
            return v
    raise ValueError(f"{kind} projection failed")


class BudgetExpired(RuntimeError):
    pass


def check_budget(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExpired("time budget reached; pool excluded and retried on resume")


def straight_search(base, cert, L, objective, band, config, lo, hi, deadline):
    """A fine rate grid, with refinements around its eight best feasible rates."""
    lr = np.log(np.arange(1, L + 1))
    power = float(np.polyfit(lr, np.log(-cert[:L]), 1)[0])
    if not power > 0:
        raise ValueError("certificate has non-positive rank power")
    s0 = 1.0 / power
    grid = s0 * np.exp(np.linspace(math.log(0.01), math.log(100.0), config["slope_points"]))
    best, best_mse, best_s = None, math.inf, None
    scores = []

    def visit(s):
        nonlocal best, best_mse, best_s
        check_budget(deadline)
        # A safe top-L line; its bulk initially follows the certificate's shift rule.
        log_shape = lr / s
        log_depth = float(np.max(np.log(-base[:L]) - log_shape))
        with np.errstate(over="ignore"):
            top = -np.exp(log_depth + log_shape)
        u = base.copy()
        u[:L] = top
        u[L:] -= base[L - 1] - top[-1]
        v = exact_scale(u, base, band, config["band"], lo, hi)
        if v is None:
            return math.inf
        try:
            value = mse(objective.errors(v))
        except ValueError:
            return math.inf
        if value < best_mse:
            best, best_mse, best_s = v, value, float(s)
        return value

    for s in grid:
        scores.append(visit(s))
    for i in np.argsort(scores)[:8]:
        if not math.isfinite(scores[i]):
            continue
        for s in np.geomspace(grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)],
                              config["refine_points"]):
            visit(s)
    # Retain the certificate's rate exactly when the line extent is unchanged.
    visit(s0)
    if best is None:
        raise ValueError("no feasible straight-MSE tail in slope search")
    return best, {"rate": best_s, "L": L, "mse": best_mse,
                  "grid_endpoint": bool(best_s <= grid[0] or best_s >= grid[-1])}


def optimize(starts, base, objective, band, config, lo, hi, deadline):
    """Monotone projected descent, retaining only exactly scaled feasible steps."""
    kind = config["band"]
    best, best_loss, winner = None, math.inf, None
    history, incumbents = [], []
    successful, total_steps = 0, 0
    tbase = torch.from_numpy(np.ascontiguousarray(base))

    def consider(v, label):
        nonlocal best, best_loss, winner
        if v is None or not feasible(v, base, band, kind, lo, hi):
            return None
        try:
            value = mse(objective.errors(v))
        except ValueError:
            return None
        if value < best_loss:
            best, best_loss, winner = v.copy(), value, label
        return value

    for label, initial in starts:
        check_budget(deadline)
        initial = np.sort(initial)[::-1]
        original_loss = consider(initial, f"start:{label}:original")
        v = exact_scale(initial, base, band, kind, lo, hi)
        value = consider(v, f"start:{label}:scaled")
        incumbents.append({"name": label, "original_mse": original_loss, "scaled_mse": value})
        if value is None:
            history.append({"start": label, "status": "infeasible_start", "steps": 0})
            continue
        z = np.log(-v)
        mean_grad, square_grad = np.zeros_like(z), np.zeros_like(z)
        accepted, state = 0, "converged"
        for step in range(config["iterations"]):
            check_budget(deadline)
            total_steps += 1
            t = torch.tensor(z, requires_grad=True)
            scaled = torch_scaled(t, tbase, band, kind, lo)
            loss = objective.torch_errors(scaled).square().mean()
            if not torch.isfinite(loss):
                state = "nonfinite_loss"
                break
            loss.backward()
            gradient = t.grad.detach().numpy()
            if not np.isfinite(gradient).all():
                state = "nonfinite_gradient"
                break
            mean_grad = 0.9 * mean_grad + 0.1 * gradient
            square_grad = 0.999 * square_grad + 0.001 * gradient ** 2
            direction = (mean_grad / (1 - 0.9 ** (step + 1))
                         / (np.sqrt(square_grad / (1 - 0.999 ** (step + 1))) + 1e-12))
            # Plain projected descent recovers if the momentum points uphill.
            directions = [direction, gradient / max(float(np.max(np.abs(gradient))), 1e-12)]
            moved = False
            for direction in directions:
                rate = config["learning_rate"]
                for _ in range(18):
                    trial_z = isotonic_regression(z - rate * direction).x
                    if np.max(np.abs(trial_z)) < 690:
                        trial = exact_scale(-np.exp(trial_z), base, band, kind, lo, hi)
                        try:
                            trial_loss = mse(objective.errors(trial)) if trial is not None else math.inf
                        except ValueError:
                            trial_loss = math.inf
                        if trial_loss < value - 1e-12 * max(1.0, value):
                            z, v, value = np.log(-trial), trial, trial_loss
                            consider(v, f"search:{label}")
                            moved = True
                            accepted += 1
                            break
                    rate *= 0.5
                if moved:
                    break
            if not moved:
                break
            state = "iteration_limit" if step + 1 == config["iterations"] else "converged"
        if not state.startswith("nonfinite"):
            successful += 1
        history.append({"start": label, "status": state, "steps": step + 1,
                        "accepted": accepted, "mse": value})
    if best is None or not successful:
        raise ValueError("optimization failed: no finite feasible completed restart")
    return best, {"winner": winner, "fallback": winner.startswith("start:"),
                  "iterations_run": total_steps, "starts": incumbents, "history": history}


def assign(ranked, base):
    out = np.empty_like(ranked)
    out[np.argsort(-base, kind="stable")] = ranked
    return out


def run_pool(entry, vb, raw, config, job_deadline):
    started = time.monotonic()
    deadline = min(job_deadline, started + config["pool_seconds"])
    M, k, e, n = (entry[name] for name in ("M", "k", "e", "U"))
    base = np.sort(vb)[::-1].copy()
    if (not np.isfinite(vb).all() or np.any(vb >= 0) or base.shape != (n,)
            or (raw is not None and (raw.shape != vb.shape or not np.isfinite(raw).all()
                                     or np.any(raw >= 0)))):
        raise ValueError("inputs must be finite, strictly negative, matching prompt arrays")
    certificate, tgt, e2, L, sref, cert_info = certify(vb, M, k)
    if e != e2:
        raise ValueError("geometry disagrees with certify")
    historical_lo, historical_hi = sref - MARGIN, sref + MARGIN
    selected_ref = sref if config["band"] == "involved" else involved_sigma_reference(
        vb, n_fit=M, fit_top_k=k, extrap_k=1, m=L)
    margin = MARGIN * selected_ref / sref
    lo, hi = selected_ref - margin, selected_ref + margin
    if not (0 < lo < hi):
        raise ValueError("this search requires a positive lower band edge")
    counts = grid.partition_counts(entry, config)
    if "grid" in entry:
        objective = Gather.streamed(n, counts["optimization"], config["opt_seed"], M, k)
        evaluation = Gather.streamed(n, counts["evaluation"], config["eval_seed"], M, k)
        # Full forecasts check the first 16 draws of each stored gather below.
        p_opt = permutations(n, 16, config["opt_seed"])
        p_eval = permutations(n, 16, config["eval_seed"])
    else:
        p_opt = permutations(n, config["opt_splits"], config["opt_seed"])
        p_eval = permutations(n, config["eval_splits"], config["eval_seed"])
        objective, evaluation = Gather(p_opt, M, k), Gather(p_eval, M, k)
    p_band = permutations(n, 200, 4321)
    p_legacy = permutations(n, 200, entry["legacy_seed"])
    band = Gather(p_band, M, k, e)
    base_order = np.argsort(-vb, kind="stable")
    base_rank = np.empty(n, dtype=np.int64)
    base_rank[base_order] = np.arange(n)
    legacy_ranks = Gather(base_rank[p_legacy], M, k)
    max_selected = max(int(max(g.fr.max(), g.dr.max())) + 1
                       for g in (objective, evaluation, band, legacy_ranks))
    # The general search has U free ranks. The straight family's shift begins
    # only after every rank read by any of the four split sets.
    straight_L = max(L, max_selected)
    checks = corner_checks()
    for label, vector in (("base", vb), ("certificate", certificate)):
        checks.append(forecast_check(label + "_legacy", vector, p_legacy, M, k))
        checks.append(forecast_check(label + "_optimization", np.sort(vector)[::-1], p_opt, M, k))
        checks.append(forecast_check(label + "_evaluation", np.sort(vector)[::-1], p_eval, M, k))
        if "grid" in entry:
            checks.extend(grid.check_gathers(label, vector, objective, evaluation, p_opt, p_eval, M, k))
    if "grid" in entry:
        checks.extend(grid.check_reference(entry, vb, certificate, p_legacy, M, k))
    np.testing.assert_allclose(band.sigma(base, "involved"), involved_sigma(vb, M, k, e),
                               atol=1e-12, rtol=1e-12)
    rows = {"base": vb, "certificate": certificate}
    trained = None
    if raw is not None:
        # Keep the historical assignment and the precise clamp-then-project order.
        st = involved_sigma(raw, M, k, e)
        if not math.isfinite(st) or st <= 0:
            raise ValueError("historical trained band clamp has invalid sigma")
        clamped = raw * (min(max(st, historical_lo), historical_hi) / st)
        historical, ok = project(clamped, vb, M, k, e, historical_lo, historical_hi)
        if not ok or not feasible(np.sort(historical)[::-1], base, band, "involved",
                                  historical_lo, historical_hi) or np.max(historical - vb) > 1e-9:
            raise ValueError("historical trained projection failed")
        trained = historical
        if config["band"] == "fitmax":
            rows["trained_certificate_band"] = historical
            trained = band_project(raw, vb, band, "fitmax", lo, hi)
        rows["trained"] = trained
    setup_seconds = time.monotonic() - started
    t0 = time.monotonic()
    ranked_cert = certificate[base_order]
    # Extrapolate the certificate line only to obtain its rate if L must grow.
    cert_line = ranked_cert.copy()
    power, intercept = np.polyfit(np.log(np.arange(1, L + 1)), np.log(-tgt), 1)
    cert_line[:straight_L] = -np.exp(intercept + power * np.log(np.arange(1, straight_L + 1)))
    straight, straight_info = straight_search(base, cert_line, straight_L, objective, band,
                                               config, lo, hi, deadline)
    straight_seconds = time.monotonic() - t0
    rows["straight_mse"] = assign(straight, vb)
    scale = lo / band.sigma(base, config["band"])
    base_start = base * scale
    # A contraction needs this minimal rigid shift to remain below every base rank.
    base_start -= max(0.0, float(np.max(base_start - base)))
    starts = [("certificate", ranked_cert), ("straight_mse", straight), ("base_band", base_start)]
    if trained is not None:
        starts.append(("trained", np.sort(trained)[::-1]))
    if config["band"] == "fitmax":
        starts.append(("certificate_fitmax_projection",
                       np.sort(band_project(certificate, vb, band, "fitmax", lo, hi))[::-1]))
    rng = np.random.default_rng(config["search_seed"])
    for j in range(config["random_starts"]):
        # Smooth perturbations give different initial curvature without fixing a basis
        # for the subsequent search, which still optimizes every rank independently.
        knots = np.linspace(0, 1, 12)
        noise = np.interp(np.linspace(0, 1, n), knots, rng.normal(0, 0.12, knots.size))
        z = isotonic_regression(np.log(-straight) + noise).x
        perturbed = -np.exp(z)
        perturbed *= lo / band.sigma(perturbed, config["band"])
        perturbed -= max(0.0, float(np.max(perturbed - base)))
        starts.append((f"perturbed_{j}", perturbed))
    t0 = time.monotonic()
    optimized, search = optimize(starts, base, objective, band, config, lo, hi, deadline)
    optimize_seconds = time.monotonic() - t0
    optimized_assigned = assign(optimized, vb)
    rows["optimized"] = optimized_assigned
    checks.append(forecast_check("optimized_legacy", optimized_assigned, p_legacy, M, k))
    checks.append(forecast_check("optimized_optimization", optimized, p_opt, M, k))
    checks.append(forecast_check("optimized_evaluation", optimized, p_eval, M, k))
    if "grid" in entry:
        checks.extend(grid.check_gathers("optimized", optimized, objective, evaluation, p_opt, p_eval, M, k))
    if not feasible(optimized, base, band, config["band"], lo, hi):
        raise ValueError("final optimized tail is infeasible")
    saved, summaries = {}, []
    for label, vector in rows.items():
        ranked = np.sort(vector)[::-1]
        errors = {"optimization": objective.errors(ranked), "evaluation": evaluation.errors(ranked),
                  "legacy": split_errors(vector, p_legacy, M, k=k)[0]}
        parts = band.parts(ranked)
        row = {"tail": label, **{mode + "_mse": mse(err) for mode, err in errors.items()},
               **{"sigma_" + part: val for part, val in parts.items()},
               "max_raise": float(np.max(vector - vb)),
               "feasible": feasible(ranked, base, band, config["band"], lo, hi)
                           and bool(np.max(vector - vb) <= 1e-9)}
        summaries.append(row)
        saved[label + "_values"] = vector
        for mode, err in errors.items():
            saved[label + "_" + mode + "_errors"] = err
    ranks = RANKS[RANKS <= n]
    psi_line = -(intercept + power * np.log(ranks))
    saved.update(sorted_values=optimized, shape_ranks=ranks,
                 psi_minus_certificate_line=-np.log(-optimized[ranks - 1]) - psi_line,
                 certificate_line_psi=psi_line)
    meta = {"schema": SCHEMA, "status": "ok", "entry": entry, "key": entry["key"],
            "config": config, "notice": NOTICE, "rows": summaries, "checks": checks,
            "sigma_ref": selected_ref, "certificate_sigma_ref": sref, "margin": margin,
            "lo": lo, "hi": hi, "certificate_L": L, "max_selected_rank": max_selected,
            "straight": straight_info, "certificate_info": cert_info, "search": search,
            "timing": {"setup_seconds": setup_seconds, "straight_seconds": straight_seconds,
                       "optimize_seconds": optimize_seconds, "wall_seconds": time.monotonic() - started}}
    if "grid" in entry:
        meta["partitions"] = counts
        meta["band_partitions"] = 200
        meta["rank_policy"] = "all U ranks free; streamed optimization/evaluation permutations"
        meta["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return meta, saved


def describe(meta, arrays, prefix="completed"):
    entry = meta["entry"]
    print(f"{prefix} {entry['run']}:{entry['step']} {entry['pool']} M{entry['M']} k{entry['k']} "
          f"band={meta['config']['band']} [{meta['lo']:.9f}, {meta['hi']:.9f}]", flush=True)
    for r in meta["rows"]:
        print(f"  {r['tail']:25s} MSE opt={r['optimization_mse']:.9f} eval={r['evaluation_mse']:.9f} "
              f"legacy={r['legacy_mse']:.9f} | sigma involved={r['sigma_involved']:.6f} "
              f"fit={r['sigma_fit']:.6f} deploy={r['sigma_deploy']:.6f} fitmax={r['sigma_fitmax']:.6f} "
              f"feasible={r['feasible']} max_raise={r['max_raise']:.3g}", flush=True)
    for check in meta["checks"]:
        print(f"  check {check['case']}: PASS max_abs={check['max_abs']:.3g}", flush=True)
    print("  optimized psi minus certificate line: " + " ".join(
        f"r{r}={d:+.4f}" for r, d in zip(arrays["shape_ranks"], arrays["psi_minus_certificate_line"])),
        flush=True)
    lookup = {r["tail"]: r for r in meta["rows"]}
    o, s = lookup["optimized"], lookup["straight_mse"]
    print(f"  shape gain over straight-MSE: opt={s['optimization_mse']/o['optimization_mse']:.4f}x "
          f"eval={s['evaluation_mse']/o['evaluation_mse']:.4f}x; "
          "a smaller evaluation gain can indicate optimization-split overfitting", flush=True)
    print(f"  winner={meta['search']['winner']} fallback={meta['search']['fallback']} "
          f"iterations={meta['search']['iterations_run']} wall={meta['timing']['wall_seconds']:.3f}s", flush=True)
    if "grid" in entry:
        print(f"  partitions={canonical(meta['partitions'])} process_peak_rss={meta['peak_rss_mib']:.1f} MiB; "
              f"band_partitions={meta['band_partitions']}; {meta['rank_policy']}", flush=True)


def selected_step(run, requested, fallback=None):
    ev = run / "snapshot_evals"
    available = sorted(int(m.group(1)) for p in ev.glob("step*_test_forecast_traces.npz")
                       if (m := re.fullmatch(r"step(\d+)_test_forecast_traces\.npz", p.name)))
    if requested == "latest":
        if not available:
            raise ValueError(f"no snapshot step in {run}")
        return available[-1], False
    step = int(requested)
    if step in available:
        return step, False
    if fallback is not None and fallback in available:
        return fallback, True
    raise ValueError(f"{run.name}: requested step {step} missing; available={available}")


def make_manifest(args, config):
    requests = list(args.specs)
    fallbacks = {}
    if args.fleet:
        for seed in (11, 17, 23):
            name = f"lqpw_pw2apsp_seed{seed}"
            requests.append(name + ":100")
            if seed != 17:
                fallbacks[name] = 70 if seed == 11 else 90
        requests.extend(f"lqdyn_v2e2ap50sp_seed{s}:latest" for s in (11, 17, 23, 29, 43))
    entries, arrays, seen = [], {}, set()
    source_identity = code_identity()

    def add(run, step, name, vb, vt, M, k, legacy_seed, step_fallback, sources, extra=None):
        if args.pool and not any(fnmatch.fnmatchcase(name, p) for p in args.pool):
            return
        identity = (run, step, name, M, k)
        if identity in seen:
            return
        seen.add(identity)
        i = len(entries)
        values = {"base": np.asarray(vb, dtype=np.float64)}
        if vt is not None:
            values["trained"] = np.asarray(vt, dtype=np.float64)
        input_checksum = digest_arrays({"sources": sources}, values)
        entry = {"run": run, "step": step, "pool": name, "M": M, "k": k, "U": len(vb),
                 "e": count_extrapolated_eval_points(M, len(vb) - M, plotting="weibull"),
                 "legacy_seed": legacy_seed, "step_fallback": step_fallback,
                 "input_checksum": input_checksum, "sources": sources}
        if extra:
            entry.update(extra)
        key_data = {"entry": entry, "config": config, "code": source_identity, "schema": SCHEMA}
        entry.update(key=hashlib.sha256(canonical(key_data).encode()).hexdigest(), index=i,
                     has_trained=vt is not None)
        entries.append(entry)
        for key, value in values.items():
            arrays[f"{i}_{key}"] = value

    for spec in requests:
        pattern, requested = spec.rsplit(":", 1)
        matches = sorted(RUNS.glob(pattern))
        if not matches:
            raise ValueError(f"run pattern matched nothing: {pattern}")
        for run in matches:
            step, fallback = selected_step(run, requested, fallbacks.get(run.name))
            ev = run / "snapshot_evals"
            bp, tp = ev / "base_test_forecast_traces.npz", ev / f"step{step:04d}_test_forecast_traces.npz"
            base, k = pools(bp)
            trained, trained_k = pools(tp)
            if k != trained_k or base.keys() != trained.keys():
                raise ValueError(f"{run.name}: base/trained geometry or pool names differ")
            for name, (vb, M) in sorted(base.items()):
                vt, Mt = trained[name]
                if Mt != M or vt.shape != vb.shape:
                    raise ValueError(f"{run.name}/{name}: prompt geometry mismatch")
                seed = int.from_bytes(name.encode(), "little") % (2 ** 32)
                add(run.name, step, name, vb, vt, M, k, seed, fallback, [str(bp), str(tp)])
    if args.bank:
        bank = pools_from_bank(smoke())
        names = [name for name, v in bank.items() if v.size == 2300]
        if len(names) != 32:
            raise ValueError(f"expected 32 bank pools of 2300; got {len(names)}")
        for i, name in enumerate(names):
            for k in (10, 50):
                add("wildchat_bank", None, name, bank[name], None, 250, k, 7 + i, False,
                    ["g_transfer_v2.pools_from_bank(smoke())", name])
    if args.grid:
        grid.add_pools(args, add)
    if not entries:
        raise ValueError("no selected pools")
    meta = {"schema": SCHEMA, "kind": "manifest", "label": args.label, "config": config,
            "code": source_identity, "entries": entries,
            "request": {"specs": args.specs, "fleet": args.fleet, "bank": args.bank, "pool": args.pool}}
    if args.grid:
        meta["request"].update(grid=True, grid_pools=args.grid_pools,
                               grid_cell=args.grid_cell, grid_k=args.grid_k)
    return meta, arrays


def validate_result(meta, arrays, entry, vb, config):
    # Pool identity is independent of its position in a particular shard manifest.
    identity = lambda e: {key: value for key, value in e.items() if key != "index"}
    if (meta["key"] != entry["key"] or identity(meta["entry"]) != identity(entry)
            or meta["config"] != config):
        raise ValueError("result identity does not match manifest")
    if meta["status"] == "failed":
        return
    if meta["status"] != "ok":
        raise ValueError("unknown result status")
    required = {"base", "certificate", "straight_mse", "optimized"}
    if entry["has_trained"]:
        required.add("trained")
    if not required.issubset({r["tail"] for r in meta["rows"]}):
        raise ValueError("result is missing baseline rows")
    if not np.array_equal(arrays["base_values"], vb):
        raise ValueError("result base differs from manifest")
    for row in meta["rows"]:
        value = arrays[row["tail"] + "_values"]
        if value.shape != vb.shape or not np.isfinite(value).all() or np.any(value >= 0):
            raise ValueError("invalid saved tail")
        for mode, count in grid.partition_counts(entry, config).items():
            errors = arrays[row["tail"] + "_" + mode + "_errors"]
            if errors.shape != (count,):
                raise ValueError("incomplete split errors")
            if not math.isclose(mse(errors), row[mode + "_mse"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("MSE does not match saved errors")
    optimized = arrays["sorted_values"]
    if not np.array_equal(assign(optimized, vb), arrays["optimized_values"]):
        raise ValueError("optimized assignment mismatch")
    band = Gather(permutations(entry["U"], 200, 4321), entry["M"], entry["k"], entry["e"])
    if not feasible(optimized, np.sort(vb)[::-1], band, config["band"], meta["lo"], meta["hi"]):
        raise ValueError("cached optimized tail is infeasible")
    if not meta["checks"] or any(not math.isfinite(c["max_abs"]) for c in meta["checks"]):
        raise ValueError("missing or invalid forecast checks")
    if "grid" in entry:
        grid.validate(meta)


def ratio(a, b):
    return float(a / b) if b > 0 else None


def aggregate(manifest, inputs, results, output_label, output_dir):
    valid, failed, invalid, pending = [], [], [], []
    for entry in manifest["entries"]:
        path = results / (entry["key"] + ".npz")
        if not path.exists():
            pending.append(entry["key"])
            continue
        try:
            meta, arrays = load_record(path)
            validate_result(meta, arrays, entry, inputs[f"{entry['index']}_base"], manifest["config"])
        except (OSError, ValueError, KeyError, AssertionError) as err:
            print(f"invalid record {path.name}: {err}", flush=True)
            invalid.append(entry["key"])
            continue
        if meta["status"] == "failed":
            failed.append(entry["key"])
        else:
            valid.append((meta, arrays))
    counts = {"expected": len(manifest["entries"]), "completed": len(valid),
              "failed": len(failed) + len(invalid), "pending": len(pending),
              "fallback": sum(int(m["search"]["fallback"]) for m, _ in valid),
              "step_fallback": sum(int(e["step_fallback"]) for e in manifest["entries"])}
    print("counts " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
    csv_rows, summaries, saved = [], [], {}
    groups = {}
    for meta, arrays in valid:
        e = meta["entry"]
        group = (e["run"], e["step"], e["M"], e["k"])
        groups.setdefault(group, []).append(meta)
        for row in meta["rows"]:
            csv_rows.append({"run": e["run"], "step": e["step"], "pool": e["pool"], "M": e["M"],
                             "k": e["k"], "band": meta["config"]["band"], **row,
                             "sigma_ref": meta["sigma_ref"], "sigma_lo": meta["lo"],
                             "sigma_hi": meta["hi"], "fallback": meta["search"]["fallback"]})
        prefix = e["key"] + "_"
        for key in ("sorted_values", "optimized_values", "shape_ranks", "psi_minus_certificate_line"):
            saved[prefix + key] = arrays[key]
    for group, records in groups.items():
        lookup = [{r["tail"]: r for r in m["rows"]} for m in records]
        common = sorted(set.intersection(*(set(r) for r in lookup)))
        for mode in MODES:
            vals = {tail: np.array([r[tail][mode + "_mse"] for r in lookup]) for tail in common}
            # Each resample selects paired pools for both tails, as in feasibility.py.
            rng = np.random.default_rng(0)
            ids = rng.integers(0, len(records), size=(2000, len(records)))
            folds = {tail: ratio(vals["base"].sum(), v.sum()) for tail, v in vals.items()}
            comparisons = {}
            for numerator, denominator in (("optimized", "certificate"), ("optimized", "straight_mse"),
                                            ("trained", "optimized")):
                if numerator not in vals:
                    continue
                a, b = vals[denominator], vals[numerator]
                denom = b[ids].sum(axis=1)
                quantiles = (np.percentile(a[ids].sum(axis=1) / denom, [2.5, 50, 97.5]).tolist()
                             if np.all(denom > 0) else None)
                comparisons[numerator + "_over_" + denominator] = {
                    "ratio": ratio(a.sum(), b.sum()), "bootstrap_95": quantiles, "resamples": 2000}
            summary = {"group": list(group), "mode": mode, "pools": len(records),
                       "folds": folds, "comparisons": comparisons}
            summaries.append(summary)
            print("pooled " + canonical(summary), flush=True)
    fields = ["run", "step", "pool", "M", "k", "band", "tail", "optimization_mse", "evaluation_mse",
              "legacy_mse", "sigma_fit", "sigma_deploy", "sigma_fitmax", "sigma_involved", "max_raise",
              "feasible", "sigma_ref", "sigma_lo", "sigma_hi", "fallback"]
    csv_file = io.StringIO(newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(csv_rows)
    stem = output_dir / ("tail_opt_" + output_label)
    atomic_bytes(stem.with_suffix(".csv"), csv_file.getvalue().encode())
    meta = {"schema": SCHEMA, "notice": NOTICE, "counts": counts, "config": manifest["config"],
            "pooled": summaries, "records": [m["entry"]["key"] for m, _ in valid],
            "failed_keys": failed, "invalid_keys": invalid, "pending_keys": pending,
            "pool_details": [m for m, _ in valid]}
    save_record(stem.with_suffix(".npz"), meta, saved)
    print(f"saved {stem}.csv and {stem}.npz", flush=True)
    if manifest["request"].get("grid"):
        grid.cell_table(manifest, valid, output_dir, output_label, atomic_bytes)
        grid.runtime_estimate(valid, manifest["config"])
    else:
        runtime_estimate(valid, manifest["config"])
    return counts


def runtime_estimate(records, config):
    if not records:
        return
    # Extrapolate measured setup/search work, keeping the geometry scaling explicit.
    estimates, default_estimates = [], []
    for meta, _ in records:
        e, t = meta["entry"], meta["timing"]
        work = e["U"] * e["k"]
        iterations = max(meta["search"]["iterations_run"], 1)
        per_step = t["optimize_seconds"] / iterations
        default_starts = 4 + config["random_starts"] + (config["band"] == "fitmax")
        search = per_step * config["iterations"] * default_starts
        # 21 password, 40 held-out WildChat, 32 bank k10, 32 bank k50; two bands.
        weighted = 2 * (21 * 935 * 10 + (40 + 32) * 2300 * 50 + 32 * 2300 * 10) / work
        estimates.append((t["setup_seconds"] + t["straight_seconds"] + search) * weighted / 3600)
        default_search = per_step * 300 * default_starts
        default_estimates.append((t["setup_seconds"] + t["straight_seconds"] + default_search) * weighted / 3600)
    print(f"runtime estimate for 250 pool/band tasks at this configuration: "
          f"{np.median(estimates):.2f} CPU-allocation wall hours serial "
          f"(observed extrapolations {min(estimates):.2f}-{max(estimates):.2f}); "
          "assumes work scales with U*k and every restart uses its iteration budget; "
          "bank loading and cluster contention are unmeasured", flush=True)
    print(f"same extrapolation at the default 300 iterations/start: "
          f"{np.median(default_estimates):.2f} serial allocation hours for 250 tasks", flush=True)


def self_test():
    started = time.monotonic()
    checks = corner_checks()
    for check in checks:
        print(f"PASS {check['case']} max_abs={check['max_abs']:.3g}")
    n, M, k, e = 47, 15, 5, 3
    rng = np.random.default_rng(51)
    base = -np.exp(np.linspace(-1, 2, n))
    shuffled = base[rng.permutation(n)]
    p = permutations(n, 20, 71)
    g = Gather(p, M, k, e)
    band = Gather(permutations(n, 200, 4321), M, k, e)
    np.testing.assert_allclose(band.sigma(base, "involved"), involved_sigma(shuffled, M, k, e))
    np.testing.assert_array_equal(assign(base, shuffled), shuffled)
    report = forecast_check("prompt_assignment", shuffled, p, M, k)
    print(f"PASS prompt_assignment max_abs={report['max_abs']:.3g}")
    lo = band.sigma(base, "involved") * 1.1
    hi = lo * 1.02
    scaled = exact_scale(base, base, band, "involved", lo, hi)
    assert feasible(scaled, base, band, "involved", lo, hi)
    assert exact_scale(-np.ones(n), base, band, "involved", lo, hi) is None
    assert exact_scale(np.full(n, np.nan), base, band, "involved", lo, hi) is None
    assert exact_scale(base[::-1], base, band, "involved", lo, hi) is None
    assert exact_scale(base, base * 20, band, "involved", lo, hi) is None
    print("PASS exact_scale: lower band, no raise, monotonicity and nonfinite/infeasible rejection")
    cfg = {"band": "involved", "iterations": 3, "learning_rate": 0.03}
    result, info = optimize([("base", scaled)], base, g, band, cfg, lo, hi, math.inf)
    assert feasible(result, base, band, "involved", lo, hi)
    assert mse(g.errors(result)) <= mse(g.errors(scaled)) + 1e-12
    print(f"PASS optimization: feasible, incumbent retained; {info['iterations_run']} iterations")
    result_dir = READOUTS / "tail_opt_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="selftest_", dir=result_dir) as folder:
        path = Path(folder) / "record.npz"
        meta, arrays = {"schema": SCHEMA, "status": "ok"}, {"v": result}
        save_record(path, meta, arrays)
        loaded, data = load_record(path)
        assert loaded == meta
        np.testing.assert_array_equal(data["v"], result)
        with open(path, "wb") as f:
            np.savez(f, metadata=canonical(dict(meta, payload_digest="wrong")), v=result)
        try:
            load_record(path)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt record was accepted")
    print(f"PASS atomic record roundtrip and corruption rejection; wall={time.monotonic()-started:.3f}s")


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="*", help="RUN_GLOB:STEP; STEP may be latest")
    ap.add_argument("--bank", action="store_true", help="32 bank pools, at both k10 and k50")
    ap.add_argument("--grid", action="store_true", help="random pools on cert_scale.CELLS; no trained tails")
    ap.add_argument("--grid-pools", type=int, default=8, help="first pools per cell, capped by cert_scale.n_pools")
    ap.add_argument("--grid-cell", action="append", default=[], help="cell name, e.g. M250_R8.2; repeatable")
    ap.add_argument("--grid-k", action="append", type=int, default=[], help="select grid k values; repeatable")
    ap.add_argument("--fleet", action="store_true",
                    help="the band arm: password seeds 11, 17, 23 at step 100 and five WildChat seeds at the latest step")
    ap.add_argument("--band", choices=("involved", "fitmax"), default="involved")
    ap.add_argument("--pool", action="append", default=[], help="pool name/glob; repeat to select several")
    ap.add_argument("--shard", default="0/1", help="zero-based I/N over the manifest's fixed pool order")
    ap.add_argument("--label", help="aggregate and default manifest label")
    ap.add_argument("--manifest", type=Path, help="reuse a frozen manifest; no run glob is needed")
    ap.add_argument("--results-dir", type=Path, default=READOUTS / "tail_opt_results")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--opt-splits", type=int, default=2000)
    ap.add_argument("--eval-splits", type=int, default=4000)
    ap.add_argument("--opt-seed", type=int, default=2026110401)
    ap.add_argument("--eval-seed", type=int, default=2026110402)
    ap.add_argument("--search-seed", type=int, default=2026110403)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--random-starts", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--slope-points", type=int, default=401)
    ap.add_argument("--refine-points", type=int, default=41)
    ap.add_argument("--pool-seconds", type=float, default=600)
    ap.add_argument("--job-seconds", type=float, default=6900, help="stop with margin before Slurm's two-hour limit")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--interrupt-after", type=int, help="SIGTERM self after N newly completed pools; resume-test hook")
    ap.add_argument("--self-test", action="store_true")
    return ap


def main(argv=None):
    started = time.monotonic()
    argv = sys.argv[1:] if argv is None else argv
    ap = parser()
    args = ap.parse_args(argv)
    explicit = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    if args.grid and (args.specs or args.bank or args.fleet):
        ap.error("--grid cannot be combined with run, bank or fleet inputs")
    if args.grid_pools < 1:
        ap.error("--grid-pools must be positive")
    if not args.grid and not args.manifest and (args.grid_cell or args.grid_k or "--grid-pools" in explicit):
        ap.error("grid selectors require --grid or a grid --manifest")
    if args.grid:
        if args.label is None:
            args.label = "grid" if args.band == "involved" else "grid_fitmax"
        if "--pool-seconds" not in explicit:
            args.pool_seconds = 3500
    if args.opt_splits < 2000 or args.eval_splits < 4000:
        ap.error("use at least 2000 optimization and 4000 independent evaluation partitions")
    if args.opt_seed == args.eval_seed or 4321 in (args.opt_seed, args.eval_seed):
        ap.error("optimization, evaluation and band seeds must differ")
    if (args.iterations < 1 or args.random_starts < 2 or args.learning_rate <= 0
            or args.slope_points < 401 or args.refine_points < 21 or args.pool_seconds <= 0
            or args.job_seconds <= 0 or args.threads < 1):
        ap.error("invalid search budget; need iterations>=1, random-starts>=2 and slope-points>=401")
    allocation = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    if args.threads > min(4, allocation):
        ap.error("threads exceeds four CPUs or the current allocation")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_test:
        self_test()
        return 0
    print(NOTICE, flush=True)
    config = {name: getattr(args, name) for name in CONFIG_NAMES}
    if args.label is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        ap.error("label must contain only letters, digits, underscores and hyphens")
    manifest_path = args.manifest or args.results_dir / ((args.label or "benchmark") + ".manifest.npz")
    if manifest_path.exists():
        manifest, inputs = load_record(manifest_path)
        if manifest.get("kind") != "manifest" or manifest["code"] != code_identity():
            ap.error("manifest code identity changed; prepare a new manifest with a new label")
        explicit = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
        for name in CONFIG_NAMES:
            if "--" + name.replace("_", "-") in explicit and config[name] != manifest["config"][name]:
                ap.error(f"--{name.replace('_', '-')} differs from frozen manifest")
        config = manifest["config"]
        grid.validate_request(args, explicit, manifest, ap)
        if args.specs or args.bank or args.fleet:
            req = manifest["request"]
            if (args.specs, args.bank, args.fleet) != (req["specs"], req["bank"], req["fleet"]):
                ap.error("input request differs from frozen manifest")
        if config["threads"] > min(4, allocation):
            ap.error("manifest threads exceeds allocation")
        torch.set_num_threads(config["threads"])
        print(f"loaded manifest {manifest_path}: steps and input arrays are frozen", flush=True)
    else:
        if args.manifest:
            ap.error("--manifest must name an existing prepared manifest")
        if args.aggregate_only:
            ap.error("cannot aggregate without an existing manifest")
        args.label = args.label or "benchmark"
        manifest, inputs = make_manifest(args, config)
        # Exclusive creation prevents two preparers from silently replacing a manifest.
        # The bytes are committed atomically; workers only open the completed path.
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        import fcntl
        with open(str(manifest_path) + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if manifest_path.exists():
                previous, old_inputs = load_record(manifest_path)
                if digest_arrays(previous, old_inputs) != digest_arrays(manifest, inputs):
                    ap.error("another preparer created a different manifest; use a new label")
            else:
                save_record(manifest_path, manifest, inputs)
        print(f"prepared manifest {manifest_path}: {len(manifest['entries'])} pools", flush=True)
    label = args.label or manifest["label"]
    if args.prepare_only:
        print("resolved steps " + canonical(sorted({(e["run"], e["step"], e["step_fallback"])
                                                   for e in manifest["entries"]})), flush=True)
        return 0
    try:
        shard, shards = map(int, args.shard.split("/"))
        if not 0 <= shard < shards:
            raise ValueError
    except ValueError:
        ap.error("--shard must have 0 <= I < N")
    results = args.results_dir / "pools"
    results.mkdir(parents=True, exist_ok=True)
    completed_now = 0
    if not args.aggregate_only:
        for entry in manifest["entries"]:
            if entry["index"] % shards != shard:
                continue
            if args.pool and not any(fnmatch.fnmatchcase(entry["pool"], p) for p in args.pool):
                continue
            if not grid.selected(entry, args):
                continue
            if time.monotonic() - started >= args.job_seconds - 5:
                print("job budget reached between pools; pending pools remain resumable", flush=True)
                break
            vb = inputs[f"{entry['index']}_base"]
            raw = inputs.get(f"{entry['index']}_trained")
            path = results / (entry["key"] + ".npz")
            if path.exists():
                try:
                    meta, arrays = load_record(path)
                    validate_result(meta, arrays, entry, vb, config)
                    if meta["status"] == "ok":
                        describe(meta, arrays, prefix="skipped validated")
                        continue
                    print(f"retry failed {entry['pool']}: {meta['error']}", flush=True)
                except (OSError, ValueError, KeyError, AssertionError) as err:
                    print(f"retry invalid {entry['pool']}: {err}", flush=True)
            print(f"starting {entry['run']}:{entry['step']} {entry['pool']} k{entry['k']}", flush=True)
            try:
                meta, arrays = run_pool(entry, vb, raw, config, started + args.job_seconds)
                validate_result(meta, arrays, entry, vb, config)
            except Exception as err:
                meta = {"schema": SCHEMA, "status": "failed", "key": entry["key"], "entry": entry,
                        "config": config, "error": f"{type(err).__name__}: {err}"}
                save_record(path, meta, {})
                print(f"failed {entry['pool']}: {meta['error']}; excluded from pooled folds", flush=True)
                continue
            save_record(path, meta, arrays)
            describe(meta, arrays)
            completed_now += 1
            if args.interrupt_after is not None and completed_now >= args.interrupt_after:
                print("test interruption: committed pool record; sending SIGTERM", flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
    # Shard-local aggregate names prevent concurrent jobs from replacing one another.
    out_label = label if shards == 1 else f"{label}_shard{shard:03d}of{shards:03d}"
    aggregation_manifest = manifest
    if manifest["request"].get("grid") and shards > 1:
        # Each worker reads only its own records; the final aggregate reads all
        # shards once, after the array finishes.
        aggregation_manifest = dict(manifest, entries=[e for e in manifest["entries"]
                                                       if e["index"] % shards == shard])
    counts = aggregate(aggregation_manifest, inputs, results, out_label, READOUTS)
    print(f"invocation wall={time.monotonic()-started:.3f}s", flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
