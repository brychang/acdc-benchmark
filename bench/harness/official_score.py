"""Score a matching with the challenge's own formula, independent of either implementation.

Transcribed from the sample code on the challenge page:

    alignment = sum over male edges (x, y) of min(EM(x, y), EF(f(x), f(y)))

Both implementations compute ``sum(min(A@P, P@B))`` instead, which is the same
number for a permutation P but shares their sparse-matrix machinery. Scoring the
two solution files this way keeps the accuracy comparison from depending on
either codebase being right.
"""

from __future__ import annotations

import argparse
import sys

from prep import read_edges, read_matching


def official_score(male_csv, female_csv, matching_csv):
    male_edges = {(a, b): w for a, b, w in read_edges(male_csv)}
    female_edges = {(a, b): w for a, b, w in read_edges(female_csv)}
    matching = dict(read_matching(matching_csv))

    missing = {x for e in male_edges for x in e} - matching.keys()
    if missing:
        raise SystemExit(f"matching is missing {len(missing)} male nodes that carry edges")

    seen = set()
    for f in matching.values():
        if f in seen:
            raise SystemExit("matching is not 1:1 - a female node is used twice")
        seen.add(f)

    total = 0
    for (x, y), w in male_edges.items():
        total += min(w, female_edges.get((matching[x], matching[y]), 0))
    return total


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--male", required=True)
    p.add_argument("--female", required=True)
    p.add_argument("--matching", required=True)
    a = p.parse_args(argv)
    print(official_score(a.male, a.female, a.matching))


if __name__ == "__main__":
    sys.exit(main())
