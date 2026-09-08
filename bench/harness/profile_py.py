"""Attribute acdc_py's runtime to individual algorithm steps.

The C++ side reports a scope breakdown through its own profiler; this produces
the comparable numbers for the Python side by wrapping the functions at their
call sites. ``permutation_match`` is split by backend, because Frank-Wolfe uses
the dense solver for the search direction and the sparse one to project the
iterate back to a permutation, and those behave very differently at scale.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

from run_py import MALE_CSV, FEMALE_CSV, WARM_CSV, read_connectome, read_matching

import scipy.sparse as sp


TOTALS = collections.defaultdict(float)
COUNTS = collections.defaultdict(int)


def _record(name, seconds):
    TOTALS[name] += seconds
    COUNTS[name] += 1


def _wrap(module, attr, name, split_solver=False):
    fn = getattr(module, attr)

    def wrapper(*args, **kwargs):
        key = name
        if split_solver:
            key = f"{name}[{kwargs.get('solver', 'dense')}]"
        t = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _record(key, time.perf_counter() - t)

    setattr(module, attr, wrapper)


def install():
    import acdc.core as core
    import acdc.frank_wolfe as fw
    import acdc.swaps as swaps

    _wrap(core, "do_frank_wolfe", "do_frank_wolfe")
    _wrap(core, "greedy_search", "greedy_search")
    _wrap(fw, "compute_gradient", "compute_gradient")
    _wrap(fw, "gradient_entries", "gradient_entries")
    _wrap(fw, "permutation_match", "permutation_match", split_solver=True)
    _wrap(swaps, "evaluate_swaps", "evaluate_swaps")
    _wrap(swaps, "make_swaps", "make_swaps")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", required=True)
    p.add_argument("--outer", type=int, default=1)
    p.add_argument("--fw", type=int, default=10)
    p.add_argument("--solver", default="dense", choices=["dense", "sparse"])
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)

    install()
    from acdc import acdc_match, score

    ar, ac, aw, an = read_connectome(os.path.join(a.csv_dir, MALE_CSV))
    br, bc, bw, bn = read_connectome(os.path.join(a.csv_dir, FEMALE_CSV))
    mi, fi = read_matching(os.path.join(a.csv_dir, WARM_CSV))
    n = max(an, bn, int(mi.max()) + 1, int(fi.max()) + 1)
    p0 = np.empty(n, dtype=np.int64)
    p0[mi] = fi

    A = sp.csr_matrix((aw, (ar, ac)), shape=(n, n))
    B = sp.csr_matrix((bw, (br, bc)), shape=(n, n))

    t = time.perf_counter()
    M = acdc_match(A, B, p0, max_iter=a.outer, num_frank_wolfe=a.fw,
                   solver=a.solver, verbose=False)
    total = time.perf_counter() - t

    rows = sorted(TOTALS.items(), key=lambda kv: -kv[1])
    print(f"n={n} outer={a.outer} fw={a.fw} solver={a.solver} "
          f"total={total:.2f}s score={score(M, A, B):.0f}")
    print(f"{'scope':<34}{'count':>8}{'total(s)':>12}{'avg(ms)':>12}{'% total':>10}")
    for name, secs in rows:
        c = COUNTS[name]
        print(f"{name:<34}{c:>8}{secs:>12.3f}{secs * 1000 / c:>12.3f}"
              f"{100 * secs / total:>9.1f}%")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"n": int(n), "outer": a.outer, "fw": a.fw, "solver": a.solver,
                       "total_s": total,
                       "scopes": {k: {"count": COUNTS[k], "total_s": v} for k, v in rows}},
                      fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
