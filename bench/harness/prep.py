"""Build reduced VNC matching instances that both implementations can read.

A subset instance keeps a random sample of the benchmark matching's node pairs
and the edges induced on them, renumbered to a contiguous 1..k so the CSVs look
exactly like the challenge files. The restricted benchmark matching stays a
valid warm start, which matters because AC/DC is a local search and its quality
depends on where it starts.

Node ids are ordered so the highest id is non-isolated in both graphs. Both
drivers then infer the same n: the C++ takes max(rows(A), rows(B), rows(P)) but
acdc_py's ``_coerce_pair`` ignores the warm start, so a trailing isolated node
would silently give them different n.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import random
import sys

MALE_CSV = "male_connectome_graph.csv"
FEMALE_CSV = "female_connectome_graph.csv"
WARM_CSV = "vnc_matching_submission_benchmark_5154247.csv"


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def read_edges(path):
    """Return [(from_id:int, to_id:int, weight:int)] with the id prefix stripped."""
    out = []
    with _open(path) as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            if len(row) < 3:
                continue
            out.append((int(row[0][1:]), int(row[1][1:]), int(row[2])))
    return out


def read_matching(path):
    """Return [(male_id:int, female_id:int)]."""
    out = []
    with _open(path) as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            out.append((int(row[0][1:]), int(row[1][1:])))
    return out


def _renumber(nodes, incident):
    """Map old id -> new 1..k, isolated nodes first so id k has at least one edge."""
    ordered = sorted(nodes, key=lambda v: (v in incident, v))
    return {old: i + 1 for i, old in enumerate(ordered)}


def build_subset(src, dst, k, seed):
    male = read_edges(os.path.join(src, MALE_CSV))
    female = read_edges(os.path.join(src, FEMALE_CSV))
    pairs = read_matching(os.path.join(src, WARM_CSV))

    if k >= len(pairs):
        raise SystemExit(f"k={k} must be smaller than the {len(pairs)} matched pairs")

    rng = random.Random(seed)
    sample = rng.sample(pairs, k)
    male_keep = {m for m, _ in sample}
    female_keep = {f for _, f in sample}

    male_sub = [e for e in male if e[0] in male_keep and e[1] in male_keep]
    female_sub = [e for e in female if e[0] in female_keep and e[1] in female_keep]

    male_inc = {e[0] for e in male_sub} | {e[1] for e in male_sub}
    female_inc = {e[0] for e in female_sub} | {e[1] for e in female_sub}
    if not male_inc or not female_inc:
        raise SystemExit(f"k={k} induced an empty subgraph; pick a larger k")

    mmap = _renumber(male_keep, male_inc)
    fmap = _renumber(female_keep, female_inc)

    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, MALE_CSV), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["From Node ID", "To Node Id", "Weight"])
        for a, b, wt in male_sub:
            w.writerow([f"m{mmap[a]}", f"m{mmap[b]}", wt])
    with open(os.path.join(dst, FEMALE_CSV), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["From Node ID", "To Node Id", "Weight"])
        for a, b, wt in female_sub:
            w.writerow([f"f{fmap[a]}", f"f{fmap[b]}", wt])
    with open(os.path.join(dst, WARM_CSV), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Male Node ID", "Female Node ID"])
        for m, f in sorted(sample, key=lambda p: mmap[p[0]]):
            w.writerow([f"m{mmap[m]}", f"f{fmap[f]}"])

    print(
        f"n={k} male_edges={len(male_sub)} female_edges={len(female_sub)} "
        f"max_male_id={max(mmap[v] for v in male_inc)} "
        f"max_female_id={max(fmap[v] for v in female_inc)} -> {dst}"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="directory with the full challenge CSVs")
    p.add_argument("--dst", required=True, help="output directory for the subset")
    p.add_argument("--n", type=int, required=True, help="number of node pairs to keep")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    build_subset(a.src, a.dst, a.n, a.seed)


if __name__ == "__main__":
    sys.exit(main())
