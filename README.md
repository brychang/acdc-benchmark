# AC/DC search: an independent benchmark of two re-implementations

This repository is a reproducible speed and accuracy benchmark of two
independent re-implementations of **AC/DC search**, the algorithm behind the
winning solution to the
[FlyWire VNC Matching Challenge](https://codex.flywire.ai/app/vnc_matching_challenge).

AC/DC alternates a **continuous** phase (Frank–Wolfe optimization over the
Birkhoff polytope of doubly stochastic matrices, with a linear-assignment
subproblem at each step) with a **discrete** phase (greedy pairwise-swap search
over permutations), each warm-starting the other. The objective is the
overlapping edge weight between the male and female VNC connectomes under a
node correspondence `p`:

```
J(p) = sum_ij min( A_ij , B_{p(i),p(j)} )
```

The algorithm is the work of Lee, Matsliah and Saul:

> D. D. Lee, A. Matsliah & L. K. Saul, *"AC/DC search: behind the winning
> solution to the FlyWire graph-matching challenge"*, Transactions on Machine
> Learning Research (01/2026).
> <https://openreview.net/forum?id=8MjCOMyaDf>

Their MATLAB reference implementation ships as supplementary material to that
paper. Nothing in this repository re-implements the algorithm; it only runs the
two existing implementations on identical inputs and scores their output.

## The two implementations

| | Language | Repository | Notes |
|---|---|---|---|
| `acdc_py` | **Python** | [flyconnectome/acdc_py](https://github.com/flyconnectome/acdc_py) | numpy / scipy, three inner loops compiled with numba; distributed on PyPI as `acdc-search`, imported as `acdc` |
| `vnc-translation` | **C++** | [brychang/vnc-translation](https://github.com/brychang/vnc-translation) | Eigen for sparse and dense numerics, OpenMP for the gradient, vendored Jonker–Volgenant LAPJV (`third_party/gatagat_lapjv`) |

Both are translations of the same MATLAB reference, done independently, and both
are MIT-licensed. Credit for the algorithm belongs to the paper's authors;
credit for the implementations belongs to their respective authors
(`acdc_py` by Philipp Schlegel, `vnc-translation` by Bryan Chang). Neither
project is vendored here — the harness expects them cloned side by side (see
[Reproducing](#reproducing)).

Disclosure: the author of this harness also wrote `vnc-translation`. That is
part of why accuracy here is measured with the challenge's own scorer applied to
the written solution files rather than with either implementation's self-report,
and why the harness, the seeded instance generator and the raw per-run logs are
all published so the numbers can be re-derived.

One difference matters for the results below. `acdc_py` deliberately drops
MATLAB's warm-start preconditioner in front of the assignment solver and calls
SciPy's exact `linear_sum_assignment`; `vnc-translation` keeps the
preconditioner and feeds a vendored LAPJV. Both are exact solvers for the same
subproblem, so they can disagree only when the assignment problem has ties, but
that is where the small residual accuracy differences come from.

## Methodology

**Identical inputs.** Both implementations read the same challenge-format CSV
directory: `male_connectome_graph.csv`, `female_connectome_graph.csv` and the
benchmark matching `vnc_matching_submission_benchmark_5154247.csv` used as the
warm start. Each run gets a fresh copy of the instance directory, because both
drivers write their solution back into it.

**Identical parameters.** `outer=5` alternations, `fw=10` Frank–Wolfe updates
per alternation, dense assignment solver, unlimited swaps, warm-started from the
benchmark matching. `harness/run_py.py` mirrors `main_acdc`'s command line
one flag at a time so that the two drivers can be pointed at one directory and
compared directly.

**Subset instances.** Sizes below the full challenge scale are built by
`harness/prep.py`, which samples `n` matched node pairs from the benchmark
matching, keeps the edges induced on those nodes, and renumbers to a contiguous
`1..n`. Sampling *pairs* rather than nodes is what keeps the restricted
benchmark matching a valid warm start — AC/DC is a local search, so its result
depends on where it starts, and an invalid or arbitrary warm start would not be
comparable to the challenge setting. Node ids are ordered so that the highest id
is non-isolated in both graphs, because the two drivers infer `n` differently
(the C++ takes `max(rows(A), rows(B), rows(P))`, `acdc_py` looks only at `A` and
`B`) and a trailing isolated node would silently put them on different problem
sizes. The sample is seeded (`--seed 0`), so instances are reproducible.

**Accuracy is measured independently.** `harness/official_score.py` is a direct
transcription of the scoring formula from the challenge page, and it is applied
to the *solution file each implementation wrote*. The number an implementation
reports about itself is never used. Both implementations internally compute
`sum(min(A@P, P@B))`, which is the same quantity for a permutation but shares
their own sparse-matrix machinery; scoring the files instead keeps the accuracy
comparison from depending on either codebase being correct. The scorer also
checks that the matching is one-to-one and covers every male node carrying an
edge.

**Timing.** Reported two ways. `algo_s` is the algorithm phases only
(Frank–Wolfe plus greedy search, taken from the C++ profiler scopes and from
equivalent wrappers installed around the same two functions in `acdc_py`);
`wall_s` is end-to-end process time, which additionally includes CSV parsing —
hand-written C++ on one side, numpy on the other — and, for Python, interpreter
and numba import. The comparison below uses `algo_s`, which is the fairer
number for the algorithm itself. Each configuration was run twice
(`--repeats 2`) and the **fastest** repeat is reported; scores are
deterministic across repeats, timings are not. Peak resident set size comes from
`/usr/bin/time -l`, which covers the whole process including the interpreter.

## Results

Apple M4 Max, best of 2 repeats, `outer=5`, `fw=10`, benchmark warm start.
Scores are from the independent challenge scorer. Full detail, including
per-phase timings and memory, is in
[`bench/results/sweep.json`](bench/results/sweep.json); the console transcript
is [`bench/results/sweep.log`](bench/results/sweep.log).

| n | warm start | C++ score | Python score | C++ algo s | Python algo s | faster |
|---|---|---|---|---|---|---|
| 500 | 3,073 | 3,543 | **3,550** | 0.262 | **0.216** | Python, 1.21x |
| 1,000 | 14,843 | **15,986** | 15,966 | 0.975 | **0.822** | Python, 1.19x |
| 2,000 | 59,731 | **63,993** | 63,907 | 3.747 | **3.601** | Python, 1.04x |
| 4,000 | 247,533 | **265,814** | 265,691 | **23.538** | 25.754 | C++, 1.09x |
| 8,000 | — | 1,062,386 | did not finish | **137.81** | — | C++, > 18x |

The n=8,000 rows appear only in `sweep.log`, not in `sweep.json`: `bench.py`
writes the JSON after each size completes, and that size aborted when the Python
run was terminated. The n=8,000 warm-start score was therefore not recorded.

### Accuracy: essentially tied

The two implementations are never more than **0.2%** apart, and the sign flips:
Python is ahead by 7 points at n=500 (+0.20%), C++ is ahead by 20 at n=1,000
(+0.13%), 86 at n=2,000 (+0.13%) and 123 at n=4,000 (+0.05%). Both improve on
the benchmark warm start by 15.3–15.5% at n=500 and 7.0–7.7% at n=1,000–4,000.
For any practical purpose the two produce the same quality of matching; the
residual gap traces to the linear-assignment step described above, not to a
difference in the search itself.

### Speed: Python wins small, C++ wins large

Below roughly n=2,000, `acdc_py` is about 20% *faster* per algorithm second than
the C++ — numba-compiled inner loops over sparse structure, with the heavy dense
work in numpy, are enough to erase the language gap at this scale. The crossover
is near n=4,000, where C++ is 9% ahead. At n=8,000 the two diverge completely.

Note that the C++ has a real advantage in end-to-end wall time at every size
that is not visible in `algo_s`: Python pays 0.16–0.21 s of interpreter and
numba import plus numpy CSV parsing before the algorithm starts, against about
1 ms to 48 ms of hand-written parsing for the C++.

### The n=8,000 finding

At n=8,000 the C++ finishes the same instance in 138 s. `acdc_py` had not
finished after 43 minutes and the run was terminated (visible as the
`command failed (-15)` line at the end of `sweep.log`). That is more than an 18x
gap, and it is a cliff rather than a continuation of the scaling trend: the
n=4,000 ratio was 1.09x.

The cause is **not** the bulk numerics. Sampling the stalled process put it
inside SciPy's sparse solver `min_weight_full_bipartite_matching` (`_lapjvsp`),
which `acdc_py` uses to project the Frank–Wolfe iterate back onto a permutation.
Two observations bound the problem:

- In the per-scope profiles ([`bench/results/prof_py_n*.json`](bench/results))
  that projection call is negligible: 0.79 ms total at n=4,000 and 1.39 ms at
  n=8,000.
- The *same* n=8,000 instance with `outer=1` completes in 29.6 s, with the
  projection still costing 1.4 ms.

So the pathology does not exist on the first alternation. It appears in a later
one, once the iterate's sparsity pattern has evolved into something the sparse
Jonker–Volgenant implementation handles badly. The `outer=1` profile also shows
where the time legitimately goes at this size — 17.6 s of Frank–Wolfe, of which
12.4 s is the dense assignment per step, and 12.0 s of greedy search — which is
consistent with the C++ and rules out a general blow-up.

This is a diagnosed but not fully root-caused finding. What is established is
the location of the stall and that it is input-pattern dependent rather than a
size effect; what is not yet established is which alternation triggers it and
what property of the pattern is responsible. Narrowing that is in progress.

### Phase split: both really run the same algorithm

Share of algorithm time spent in Frank–Wolfe (the remainder is greedy search):

| n | C++ | Python |
|---|---|---|
| 500 | 78% | 80% |
| 1,000 | 77% | 74% |
| 2,000 | 72% | 80% |
| 4,000 | 62% | 60% |

The Frank–Wolfe share declines with size in both, as the greedy swap phase grows
faster than the continuous phase, and the two implementations track each other
to within a few points at every size. Combined with the near-identical scores,
this is good evidence that both are executing the same algorithm and not merely
arriving at similar answers.

### Memory: provisional

Peak RSS currently favours Python at the larger sizes — 854 MB against 1,344 MB
at n=4,000 — but the readings are noisy enough that they should not be quoted.
The clearest sign of that is non-monotonicity: the C++ measured 834 MB at
n=1,000 and only 344 MB at n=2,000 on the same instance family. Both
implementations hold dense `n x n` gradient and swap-gain matrices, so both are
`O(n^2)` in principle; establishing the constant needs a measurement method
better than sampled peak RSS.

### Full challenge scale (n=18,524): pending

The full-scale run has not completed and no numbers are reported for it. At
challenge scale the dense `n x n` working matrices alone need well over 10 GB,
so the result may be dominated by memory behaviour rather than by the arithmetic
measured above. This section will be filled in when that run finishes.

## Reproducing

Clone the two implementations next to the harness, inside `bench/`:

```bash
git clone https://github.com/flyconnectome/acdc_py            bench/acdc_py
git clone https://github.com/brychang/vnc-translation         bench/vnc-translation
```

Build the C++ driver. The harness looks for the binary at
`bench/vnc-translation/src/build/main_acdc`, so configure the build tree there
(upstream's `src/build.sh` places it at the repository root instead):

```bash
brew install cmake eigen libomp          # macOS; OpenMP is optional
mkdir -p bench/vnc-translation/src/build
cmake -S bench/vnc-translation/src -B bench/vnc-translation/src/build \
      -DCMAKE_BUILD_TYPE=Release
cmake --build bench/vnc-translation/src/build --parallel
```

Install the Python implementation into a virtualenv at `bench/.venv`, which is
the interpreter the harness invokes:

```bash
python3 -m venv bench/.venv
bench/.venv/bin/pip install -e bench/acdc_py
```

Fetch the challenge data (about 119 MB decompressed; not redistributed here):

```bash
scripts/fetch_data.sh                    # writes bench/data/
```

Run the sweep. Instances are generated on first use and cached in
`bench/instances/`:

```bash
cd bench
./.venv/bin/python harness/bench.py \
    --sizes 500 1000 2000 4000 8000 \
    --outer 5 --fw 10 --repeats 2 \
    --out results/sweep.json
```

Add `--full` to include the untouched n=18,524 challenge instance. To reproduce
the per-scope Python profile that located the n=8,000 stall:

```bash
cd bench
./.venv/bin/python harness/profile_py.py \
    --csv-dir instances/n8000 --outer 1 --fw 10 \
    --json results/prof_py_n8000.json
```

Individual pieces are usable on their own: `harness/prep.py` builds a subset
instance, `harness/run_py.py` is a standalone `acdc_py` driver with
`main_acdc`'s interface, and `harness/official_score.py` scores any
`(male, female, matching)` CSV triple.

## Environment the numbers were taken on

- Apple M4 Max, 16 cores (12 performance), 64 GB RAM, macOS 15.7.7 (24G720)
- Python 3.14.7, numpy 2.5.3, scipy 1.18.1, numba 0.67.0
- Apple clang 17.0.0 (arm64), CMake Release build, Eigen and Homebrew libomp

Absolute timings are specific to this machine. The cross-implementation ratios
should carry over more readily than the seconds do, but the n=8,000 stall in
particular depends on a specific SciPy version's sparse assignment code and is
worth re-checking against whatever SciPy a reader has.

## Repository contents

```
bench/harness/          the benchmark code
  prep.py               build reduced instances from the challenge CSVs
  run_py.py             acdc_py driver with main_acdc's command line
  official_score.py     the challenge's own scoring formula, applied to files
  bench.py              run both implementations and compare
  profile_py.py         per-scope attribution of acdc_py's runtime
bench/results/          measurements (timings and scores only, no graph data)
  sweep.json            the results table above, in full detail
  sweep.log             console transcript of the sweep
  results.json          an earlier partial pass (n=500, n=1,000)
  prof_py_n*.json       per-scope Python profiles at n=2,000/4,000/8,000
  logs/                 per-run stdout and /usr/bin/time -l captures
scripts/fetch_data.sh   download the challenge inputs
```

Not in the repository, and gitignored: `bench/data/`, `bench/instances/` and
`bench/work/` (challenge data and everything derived from it), `bench/.venv/`,
and the two upstream clones.

## Licensing and data

The harness code in this repository is MIT-licensed, Copyright (c) 2026 Bryan
Chang — see [LICENSE](LICENSE).

The two benchmarked projects are separate MIT-licensed repositories, and the
MATLAB reference implementation they both translate is MIT-licensed, Copyright
(c) 2025 Daniel Lee and Lawrence Saul. Consult each repository for its
authoritative licence text. If you use any of this, cite the TMLR paper: the
algorithm is its authors' work.

The challenge data is **not** redistributed here, in any form, including derived
subsets. The challenge page states: "Much of the data used in this challenge is
unpublished. Please do not publish or redistribute without first contacting
flywire@princeton.edu." `scripts/fetch_data.sh` downloads it from the challenge
endpoints so that each user obtains it under those terms directly.
