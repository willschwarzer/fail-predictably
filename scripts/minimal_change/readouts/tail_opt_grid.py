"""Grid inputs and summaries for tail_opt; the certificate remains unchanged.

The default grid has 21 cell/k combinations, hence 168 records per band at
eight pools. Pool names include the cell so --pool can select a single draw.
Optimization uses at most 2,000,000 selected fit values per objective, with a
256-partition minimum. Independent evaluation always keeps its requested count.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import os
from pathlib import Path

import numpy as np

from cert_scale import CELLS, S, n_pools, pools_from_bank, smoke
from poolscale_readout import split_errors

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[3]))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", REPO_ROOT / "artifacts/results"))
READOUTS = RESULTS_ROOT / "lquant_dynamics/readouts"


def partition_counts(entry, config):
    opt = config["opt_splits"]
    if "grid" in entry:
        # 2,000,000 selected fit values keeps all 2,000 default partitions up to k = 1000. The first
        # cap (200,000 values, 256 partitions at k1000) let the M4000 k1000 smoke fit its 256
        # partitions 8.2x better than the straight tail while 4,000 independent partitions showed
        # 1.18x, with 85% of the squared error in the worst 1% of them.
        opt = min(opt, max(256, 2_000_000 // entry["k"]))
    return {"optimization": opt, "evaluation": config["eval_splits"], "legacy": S}


def selected(entry, args):
    if not (args.grid_cell or args.grid_k):
        return True
    return ("grid" in entry
            and (not args.grid_cell or entry["grid"]["cell"] in args.grid_cell)
            and (not args.grid_k or entry["k"] in args.grid_k))


def add_pools(args, add):
    cells = {f"M{M}_R{R:g}": (M, R, ks) for M, R, ks in CELLS}
    if set(args.grid_cell) - cells.keys():
        raise ValueError(f"unknown grid cells: {sorted(set(args.grid_cell) - cells.keys())}")
    chosen = [v for cell, v in cells.items() if not args.grid_cell or cell in args.grid_cell]
    available_k = {k for _, _, ks in chosen for k in ks}
    if set(args.grid_k) - available_k:
        raise ValueError(f"k not in selected cells: {sorted(set(args.grid_k) - available_k)}")
    reference_path = READOUTS / "cert_scale.csv"
    reference_bytes = reference_path.read_bytes()
    reference_hash = hashlib.sha256(reference_bytes).hexdigest()
    references = {}
    for row in csv.DictReader(io.StringIO(reference_bytes.decode())):
        if row["source"] == "random":
            key = (row["cell"], int(row["k"]), row["pool"])
            if key in references:
                raise ValueError(f"duplicate certificate reference: {key}")
            references[key] = row
    bank = pools_from_bank(smoke())
    banks = [v for v in bank.values() if v.size == 2300]
    if len(banks) != 32:
        raise ValueError(f"expected 32 bank pools of 2300; got {len(banks)}")
    # Preserve loader insertion order, exactly as cert_scale.main does.
    population = np.concatenate(banks)
    for cell, (M, R, ks) in cells.items():
        if args.grid_cell and cell not in args.grid_cell:
            continue
        N = int(round(R * M))
        U = M + N
        if U > population.size:
            raise ValueError(f"{cell}: U={U} exceeds population of {population.size}")
        for pool_id in range(min(args.grid_pools, n_pools(U))):
            rng = np.random.default_rng([M, N, pool_id, 20261104])
            vb = population[rng.choice(population.size, size=U, replace=False)]
            pool = f"rand{pool_id:02d}"
            for k in ks:
                if args.grid_k and k not in args.grid_k:
                    continue
                ref = references.get((cell, k, pool))
                if ref is None or ref["status"] != "ok":
                    raise ValueError(f"missing successful certificate reference: {cell} k{k} {pool}")
                if (int(ref["M"]), int(ref["N"]), int(ref["U"])) != (M, N, U):
                    raise ValueError(f"certificate geometry mismatch: {cell} {pool}")
                detail = {"cell": cell, "pool_id": pool_id, "N": N, "R": N / M,
                          "reference_sha256": reference_hash,
                          "reference_base_mse": float(ref["base_mse"]),
                          "reference_cert_mse": float(ref["cert_mse"])}
                add("wildchat_grid_" + cell, None, cell + "_" + pool, vb, None, M, k,
                    7 + pool_id, False,
                    ["g_transfer_v2.pools_from_bank(smoke())", "cert_scale random draw",
                     str(reference_path)], extra={"grid": detail})


def validate_request(args, explicit, manifest, ap):
    req = manifest["request"]
    is_grid = req.get("grid", False)
    if args.grid and not is_grid:
        ap.error("--grid differs from frozen manifest")
    if is_grid and (args.specs or args.bank or args.fleet):
        ap.error("grid manifest cannot be combined with other input modes")
    if "--grid-pools" in explicit and args.grid_pools != req.get("grid_pools"):
        ap.error("--grid-pools differs from frozen manifest")
    if args.grid_cell or args.grid_k:
        if not is_grid:
            ap.error("grid selectors require a grid manifest")
        if not any(selected(e, args) for e in manifest["entries"]):
            ap.error("grid selectors match no entries in the manifest")


def check_reference(entry, base, cert, perms, M, k):
    checks = []
    for label, vector, field in (("base", base, "reference_base_mse"),
                                  ("certificate", cert, "reference_cert_mse")):
        value = float(np.mean(split_errors(vector, perms, M, k=k)[0] ** 2))
        expected = entry["grid"][field]
        np.testing.assert_allclose(value, expected, rtol=1e-10, atol=1e-10,
                                   err_msg=f"{label}: scoring MSE differs from cert_scale.csv")
        checks.append({"case": label + "_cert_scale_csv", "max_abs": abs(value - expected)})
    return checks


def check_gathers(label, vector, objective, evaluation, p_opt, p_eval, M, k):
    checks = []
    ranked = np.sort(vector)[::-1]
    for mode, gather, perms in (("optimization", objective, p_opt), ("evaluation", evaluation, p_eval)):
        actual = gather.errors(ranked)[:len(perms)]
        expected = split_errors(ranked, perms, M, k=k)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        checks.append({"case": label + "_streamed_" + mode, "max_abs": float(np.max(abs(actual - expected)))})
    return checks


def validate(meta):
    expected = partition_counts(meta["entry"], meta["config"])
    if meta.get("partitions") != expected:
        raise ValueError("grid partition counts differ from the recorded policy")
    if meta.get("band_partitions") != 200:
        raise ValueError("grid band must retain all 200 historical partitions")
    rows = {r["tail"]: r for r in meta["rows"]}
    for tail, key in (("base", "reference_base_mse"), ("certificate", "reference_cert_mse")):
        np.testing.assert_allclose(rows[tail]["legacy_mse"], meta["entry"]["grid"][key],
                                   rtol=1e-10, atol=1e-10)
    if not math.isfinite(meta["peak_rss_mib"]) or meta["peak_rss_mib"] <= 0:
        raise ValueError("missing process peak memory measurement")


def cell_rows(manifest, records):
    groups = {}
    for entry in manifest["entries"]:
        if "grid" in entry:
            key = (entry["grid"]["cell"], entry["k"])
            groups.setdefault(key, {"entry": entry, "expected": 0, "records": []})["expected"] += 1
    for meta, _ in records:
        entry = meta["entry"]
        if "grid" in entry:
            groups[(entry["grid"]["cell"], entry["k"])]["records"].append(meta)
    rows = []
    for (cell, k), group in groups.items():
        e, metas = group["entry"], group["records"]
        row = {"cell": cell, "M": e["M"], "N": e["grid"]["N"], "R": e["grid"]["R"],
               "U": e["U"], "k": k, "band": manifest["config"]["band"],
               "expected_pools": group["expected"], "completed_pools": len(metas),
               "missing_pools": group["expected"] - len(metas)}
        lookup = [{r["tail"]: r for r in m["rows"]} for m in metas]
        for mode in ("evaluation", "legacy"):
            means = {}
            for tail in ("base", "certificate", "optimized"):
                values = np.array([r[tail][mode + "_mse"] for r in lookup])
                sigmas = np.array([r[tail]["sigma_involved"] for r in lookup])
                g = values / (sigmas / math.log(10.0)) ** 2
                means[tail] = float(values.mean()) if len(values) else None
                row[f"{mode}_{tail}_mse"] = means[tail]
                row[f"{mode}_{tail}_G_mean"] = float(g.mean()) if len(g) else None
                # Population CV over the observed pools; one pool has CV zero.
                row[f"{mode}_{tail}_G_cv"] = float(g.std(ddof=0) / g.mean()) if len(g) and g.mean() > 0 else None
            for numerator, denominator in (("certificate", "base"), ("optimized", "base"),
                                             ("optimized", "certificate")):
                a, b = means[denominator], means[numerator]
                row[f"{mode}_{numerator}_over_{denominator}"] = a / b if b is not None and b > 0 else None
        rows.append(row)
    return rows


def cell_table(manifest, records, output_dir, label, atomic_bytes):
    rows = cell_rows(manifest, records)
    if not rows:
        return
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    # Labels separate bands and shards. The default involved label gives the
    # requested tail_opt_grid_cells.csv; fitmax gives tail_opt_grid_fitmax_cells.csv.
    path = output_dir / f"tail_opt_{label}_cells.csv"
    atomic_bytes(path, buf.getvalue().encode())
    print("grid cell table (MSE in decades^2; G uses each tail's involved-set sigma):", flush=True)
    print(buf.getvalue(), end="", flush=True)
    print(f"saved {path}", flush=True)


def runtime_estimate(records, config):
    tasks = [(M, int(round(M * R)), k) for M, R, ks in CELLS for k in ks]
    hours = 2 * 8 * len(tasks) * config["pool_seconds"] / 3600
    print(f"grid budget: {8 * len(tasks)} tasks/band, {2 * 8 * len(tasks)} for both bands; "
          f"{hours:.1f} serial CPU hours of pool budgets (setup may overrun a deadline)", flush=True)
    if not records:
        return
    # This is an extrapolation, not a timing guarantee. Rank work and gathered
    # fit values account for the two dimensions that increase search cost.
    def work(U, k):
        count = partition_counts({"grid": {}, "k": k}, config)["optimization"]
        return U + count * k
    rates = [m["timing"]["wall_seconds"] / work(m["entry"]["U"], m["entry"]["k"]) for m, _ in records]
    # Fitmax adds the projected certificate start to the involved band's starts.
    involved_starts = 3 + config["random_starts"]
    measured_starts = involved_starts + (config["band"] == "fitmax")
    total = 8 * (2 * involved_starts + 1) / measured_starts * sum(work(M + N, k) for M, N, k in tasks)
    print(f"grid measured-cost extrapolation, both bands: {total * np.median(rates) / 3600:.1f} "
          f"serial CPU hours (range {total * min(rates) / 3600:.1f}-{total * max(rates) / 3600:.1f}); "
          "assumes comparable restart lengths; fitmax adds one restart. "
          f"Largest observed process peak: {max(m['peak_rss_mib'] for m, _ in records):.1f} MiB; "
          "includes imports/preparation and earlier pools in the process.", flush=True)
