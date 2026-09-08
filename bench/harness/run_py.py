"""Run acdc_py over a challenge-format csv directory, mirroring main_acdc's CLI.

Same inputs, same defaults and same output file format as the C++ driver, so the
two can be pointed at one directory and compared directly. Reports the phase
timings as JSON on stdout; the algorithm time excludes CSV parsing, which is
harness code on this side and hand-written C++ on the other.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import scipy.sparse as sp

MALE_CSV = "male_connectome_graph.csv"
FEMALE_CSV = "female_connectome_graph.csv"
WARM_CSV = "vnc_matching_submission_benchmark_5154247.csv"


def _numeric_columns(path, ncols):
    """Parse a prefixed-id CSV into an (rows, ncols) float array.

    The ids carry a single alphabetic prefix ('m'/'f'), so dropping those bytes
    from the body leaves plain numeric CSV that numpy can read in one pass.
    Parsing 4.2M rows a field at a time in Python costs more than the match.
    """
    with open(path, "rb") as fh:
        buf = fh.read()
    nl = buf.find(b"\n")
    body = buf[nl + 1:] if nl >= 0 else b""
    body = body.translate(None, b"mfMF \r").strip(b"\n,")
    if not body:
        return np.zeros((0, ncols))
    flat = np.fromstring(body.replace(b"\n", b","), dtype=float, sep=",")
    if flat.size % ncols:
        raise SystemExit(f"{path}: {flat.size} values is not a multiple of {ncols}")
    return flat.reshape(-1, ncols)


def read_connectome(path):
    """Read a connectome CSV into a COO triplet list with 0-based ids."""
    a = _numeric_columns(path, 3)
    rows = a[:, 0].astype(np.int64) - 1
    cols = a[:, 1].astype(np.int64) - 1
    n = int(max(rows.max(), cols.max())) + 1 if a.shape[0] else 0
    return rows, cols, a[:, 2], n


def read_matching(path):
    a = _numeric_columns(path, 2)
    return a[:, 0].astype(np.int64) - 1, a[:, 1].astype(np.int64) - 1


def save_solution(path, perm):
    """Write the matching in the challenge's CSV format, male id ascending."""
    with open(path, "w") as fh:
        fh.write("Male Node ID,Female Node ID\n")
        fh.writelines(f"m{i + 1},f{j + 1}\n" for i, j in enumerate(perm))


def _instrument_phases():
    """Accumulate wall time per AC/DC phase into the returned dict.

    ``acdc.core`` binds ``do_frank_wolfe`` and ``greedy_search`` at import, so
    the wrappers have to replace the names in that module, not in the ones they
    are defined in.
    """
    import acdc.core as core

    totals = {"frank_wolfe_s": 0.0, "greedy_s": 0.0}

    def timed(fn, key):
        def wrapper(*args, **kwargs):
            t = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                totals[key] += time.perf_counter() - t
        return wrapper

    core.do_frank_wolfe = timed(core.do_frank_wolfe, "frank_wolfe_s")
    core.greedy_search = timed(core.greedy_search, "greedy_s")
    return totals


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", required=True)
    p.add_argument("--outer", type=int, default=5, help="AC/DC alternations (main_acdc --outer)")
    p.add_argument("--fw", type=int, default=10, help="Frank-Wolfe updates per alternation")
    p.add_argument("--max-swap", type=float, default=float("inf"))
    p.add_argument("--solver", default="dense", choices=["dense", "sparse"])
    p.add_argument("--phase", default="acdc", choices=["acdc", "continuous", "discrete"])
    p.add_argument("--warm-start-csv", default=WARM_CSV)
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--out", default=None, help="where to write the solution CSV")
    p.add_argument("--json", default=None, help="where to write the timing JSON")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)

    # Imported here so module import time lands in the load phase, not at parse.
    t0 = time.perf_counter()
    from acdc import acdc_match, frank_wolfe_search, greedy_match, score, matrix_to_perm
    import_s = time.perf_counter() - t0

    # main_acdc times its two phases separately (main_acdc.iteration.frank_wolfe /
    # .greedy_search); wrap the same two calls here so the split is comparable.
    phase_s = _instrument_phases()

    t0 = time.perf_counter()
    ar, ac, aw, an = read_connectome(os.path.join(a.csv_dir, MALE_CSV))
    br, bc, bw, bn = read_connectome(os.path.join(a.csv_dir, FEMALE_CSV))

    if a.no_warm_start:
        n = max(an, bn)
        p0 = None
    else:
        mi, fi = read_matching(os.path.join(a.csv_dir, a.warm_start_csv))
        # main_acdc sizes from max(rows(A), rows(B), rows(P)); acdc_py's
        # _coerce_pair looks only at A and B, so fold the warm start in here to
        # keep both drivers on the same n.
        n = max(an, bn, int(mi.max()) + 1, int(fi.max()) + 1)
        p0 = np.empty(n, dtype=np.int64)
        p0[:] = -1
        p0[mi] = fi
        if (p0 < 0).any():
            raise SystemExit(f"warm start does not cover all {n} nodes")

    A = sp.csr_matrix((aw, (ar, ac)), shape=(n, n))
    B = sp.csr_matrix((bw, (br, bc)), shape=(n, n))
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if a.phase == "acdc":
        M = acdc_match(A, B, p0, max_iter=a.outer, num_frank_wolfe=a.fw,
                       max_swap=a.max_swap, solver=a.solver, verbose=a.verbose)
    elif a.phase == "continuous":
        M = frank_wolfe_search(A, B, p0, num_updates=a.fw, solver=a.solver, verbose=a.verbose)
    else:
        M = greedy_match(A, B, p0, max_swap=a.max_swap, verbose=a.verbose)
    algo_s = time.perf_counter() - t0

    final = float(score(M, A, B))
    perm = matrix_to_perm(M)

    out_path = a.out or os.path.join(a.csv_dir, f"vnc_matching_py_{int(round(final))}.csv")
    t0 = time.perf_counter()
    save_solution(out_path, perm)
    save_s = time.perf_counter() - t0

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # bytes on macOS
    result = {
        "impl": "acdc_py",
        "n": int(n),
        "male_edges": int(len(aw)),
        "female_edges": int(len(bw)),
        "phase": a.phase,
        "outer": a.outer,
        "fw": a.fw,
        "solver": a.solver,
        "score": final,
        "import_s": round(import_s, 3),
        "load_s": round(load_s, 3),
        "algo_s": round(algo_s, 3),
        "frank_wolfe_s": round(phase_s["frank_wolfe_s"], 3),
        "greedy_s": round(phase_s["greedy_s"], 3),
        "save_s": round(save_s, 3),
        "peak_rss_mb": round(peak_kb / 1024.0, 1),
        "solution": out_path,
    }
    print(json.dumps(result))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
