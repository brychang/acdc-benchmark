"""Run acdc_py and vnc-translation on identical instances and compare them.

Each implementation gets its own copy of the instance directory (both drivers
write their solution back into the csv dir, and the C++ one names the file after
the score it believes it got). Timing is reported two ways: end-to-end process
wall clock, and the algorithm phases alone, since the C++ hand-rolls its CSV
parsing while the Python side goes through numpy. Accuracy is always the
independent challenge scorer applied to the written solution file, never the
number an implementation reports about itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

from official_score import official_score
from prep import MALE_CSV, FEMALE_CSV, WARM_CSV, build_subset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
CPP = os.path.join(ROOT, "vnc-translation", "src", "build", "main_acdc")

PROFILE_RE = re.compile(r"^\[profile\]\s+(\S+)\s+(\d+)\s+([\d.]+)\s")


def _maxrss_mb(path):
    """Peak RSS in MB from a `/usr/bin/time -l` stderr capture."""
    with open(path) as fh:
        m = re.search(r"(\d+)\s+maximum resident set size", fh.read())
    return round(int(m.group(1)) / 1024.0 / 1024.0, 1) if m else None


def _run(cmd, log_path):
    """Run under /usr/bin/time -l, returning (wall_s, stdout, rss_mb)."""
    rss_path = log_path + ".rss"
    t = time.perf_counter()
    proc = subprocess.run(["/usr/bin/time", "-l"] + cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t
    with open(log_path, "w") as fh:
        fh.write(proc.stdout)
    with open(rss_path, "w") as fh:
        fh.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return wall, proc.stdout, _maxrss_mb(rss_path)


def _fresh_instance(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for name in (MALE_CSV, FEMALE_CSV, WARM_CSV):
        shutil.copy(os.path.join(src, name), os.path.join(dst, name))


def run_cpp(inst, outer, fw, log_path):
    wall, out, rss = _run([CPP, "--csv-dir", inst, "--outer", str(outer), "--fw", str(fw)], log_path)

    prof = {}
    for line in out.splitlines():
        m = PROFILE_RE.match(line)
        if m:
            prof[m.group(1)] = float(m.group(3))

    # main_acdc names the file after the score it computed, so match on the
    # prefix and take the newest rather than trying to predict the name.
    sols = [os.path.join(inst, f) for f in os.listdir(inst)
            if f.startswith("vnc_matching_submission_") and f != WARM_CSV]
    if not sols:
        raise SystemExit(f"C++ wrote no solution into {inst}")
    solution = max(sols, key=os.path.getmtime)

    fw_s = prof.get("main_acdc.iteration.frank_wolfe", 0.0)
    greedy_s = prof.get("main_acdc.iteration.greedy_search", 0.0)
    return {
        "impl": "vnc-translation (C++)",
        "wall_s": round(wall, 3),
        "algo_s": round(fw_s + greedy_s, 3),
        "frank_wolfe_s": round(fw_s, 3),
        "greedy_s": round(greedy_s, 3),
        "load_s": round(prof.get("main_acdc.load_connectomes", 0.0)
                        + prof.get("main_acdc.load_solution", 0.0), 3),
        "peak_rss_mb": rss,
        "solution": solution,
    }


def run_py(inst, outer, fw, log_path):
    out_csv = os.path.join(inst, "solution_py.csv")
    wall, out, rss = _run(
        [PYTHON, os.path.join(HERE, "run_py.py"), "--csv-dir", inst,
         "--outer", str(outer), "--fw", str(fw), "--out", out_csv],
        log_path,
    )
    data = json.loads([l for l in out.splitlines() if l.startswith("{")][-1])
    return {
        "impl": "acdc_py (Python)",
        "wall_s": round(wall, 3),
        "algo_s": data["algo_s"],
        "frank_wolfe_s": data["frank_wolfe_s"],
        "greedy_s": data["greedy_s"],
        "load_s": round(data["load_s"] + data["import_s"], 3),
        # ru_maxrss is bytes on macOS; run_py already converted, but /usr/bin/time
        # covers the interpreter too, so prefer it when available.
        "peak_rss_mb": rss if rss is not None else data["peak_rss_mb"],
        "solution": data["solution"],
        "n": data["n"],
        "male_edges": data["male_edges"],
        "female_edges": data["female_edges"],
    }


def bench_size(src_inst, work, outer, fw, repeats, logdir, label):
    male = os.path.join(src_inst, MALE_CSV)
    female = os.path.join(src_inst, FEMALE_CSV)
    warm = official_score(male, female, os.path.join(src_inst, WARM_CSV))

    rows = []
    for impl, runner in (("cpp", run_cpp), ("py", run_py)):
        best = None
        for r in range(repeats):
            inst = os.path.join(work, f"{label}_{impl}")
            _fresh_instance(src_inst, inst)
            log = os.path.join(logdir, f"{label}_{impl}_r{r}.log")
            res = runner(inst, outer, fw, log)
            res["score"] = official_score(male, female, res["solution"])
            res["warm_start_score"] = warm
            res["gain_pct"] = round(100.0 * (res["score"] - warm) / warm, 3)
            res["label"] = label
            res["repeat"] = r
            print(f"  [{label}] {res['impl']:<22} run {r + 1}/{repeats}  "
                  f"algo={res['algo_s']:>9.2f}s  wall={res['wall_s']:>9.2f}s  "
                  f"score={res['score']}", flush=True)
            # Keep the fastest repeat: score is deterministic, timing is not.
            if best is None or res["algo_s"] < best["algo_s"]:
                best = res
        rows.append(best)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=os.path.join(ROOT, "data"))
    p.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    p.add_argument("--full", action="store_true", help="also run the untouched challenge instance")
    p.add_argument("--outer", type=int, default=5)
    p.add_argument("--fw", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "results.json"))
    a = p.parse_args(argv)

    instances = os.path.join(ROOT, "instances")
    work = os.path.join(ROOT, "work")
    logdir = os.path.join(ROOT, "results", "logs")
    for d in (instances, work, logdir, os.path.dirname(a.out)):
        os.makedirs(d, exist_ok=True)

    results = []
    for n in a.sizes:
        src = os.path.join(instances, f"n{n}")
        if not os.path.exists(os.path.join(src, WARM_CSV)):
            build_subset(a.data, src, n, a.seed)
        print(f"n={n}:", flush=True)
        results += bench_size(src, work, a.outer, a.fw, a.repeats, logdir, f"n{n}")
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=2)

    if a.full:
        print("n=18524 (full challenge instance):", flush=True)
        results += bench_size(a.data, work, a.outer, a.fw, 1, logdir, "full")
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=2)

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
