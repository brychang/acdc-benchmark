"""Trace every permutation_match call so the stalling one can be identified.

acdc_py finishes one AC/DC alternation at n=8000 in ~30s but does not finish two,
and a stack sample lands in scipy's sparse matcher. Frank-Wolfe calls
permutation_match twice per phase for different reasons: the dense solver picks
the search direction from the gradient, the sparse one projects the doubly
stochastic iterate back onto the permutations. This prints each call as it
returns, with the shape of the matrix that went in, so the two can be told apart.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

from run_py import MALE_CSV, FEMALE_CSV, WARM_CSV, read_connectome, read_matching


def install(dump_dir=None):
    import acdc.frank_wolfe as fw

    original = fw.permutation_match
    state = {"i": 0}

    def wrapper(W, solver="dense"):
        state["i"] += 1
        i = state["i"]
        if sp.issparse(W):
            nnz, distinct = W.nnz, len(np.unique(np.round(W.data, 12)))
            kind = f"sparse-input nnz={nnz} distinct_values={distinct}"
        else:
            kind = f"dense-input shape={W.shape}"
        print(f"[{i:03d}] permutation_match(solver={solver}) {kind} ...", flush=True)

        # Persist the input of each sparse projection: if the next line never
        # prints, this file is the reproducer for the stall.
        if dump_dir and solver == "sparse" and sp.issparse(W):
            sp.save_npz(os.path.join(dump_dir, f"proj_input_{i:03d}.npz"), sp.csr_matrix(W))

        t = time.perf_counter()
        out = original(W, solver=solver)
        el = time.perf_counter() - t
        print(f"[{i:03d}] permutation_match(solver={solver}) done in {el:.3f}s", flush=True)
        return out

    fw.permutation_match = wrapper


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", required=True)
    p.add_argument("--outer", type=int, default=2)
    p.add_argument("--fw", type=int, default=10)
    p.add_argument("--dump-dir", default=None)
    a = p.parse_args(argv)

    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)
    install(a.dump_dir)
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
    M = acdc_match(A, B, p0, max_iter=a.outer, num_frank_wolfe=a.fw, verbose=False)
    print(f"total={time.perf_counter() - t:.2f}s score={score(M, A, B):.0f}")


if __name__ == "__main__":
    sys.exit(main())
