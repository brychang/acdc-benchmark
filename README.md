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
number for the algorithm itself. Subset configurations were run twice
(`--repeats 2`) and the **fastest** repeat is reported; scores are deterministic
across repeats, timings are not. The two full-scale runs take about half an hour
each and were run once. Peak resident set size comes from `/usr/bin/time -l`,
which covers the whole process including the interpreter; where that capture is
missing the table says so rather than substituting a self-reported figure.

## Results

Apple M4 Max, `outer=5`, `fw=10`, benchmark warm start. Scores are from the
independent challenge scorer applied to the solution file each implementation
wrote. The n=500–4,000 rows are the best of 2 repeats; the n=8,000 seed rows and
the full-scale rows are single runs. Full detail, including per-phase timings
and memory, is in [`bench/results/sweep.json`](bench/results/sweep.json)
(n=500–4,000), [`bench/results/n8000_seeds.json`](bench/results/n8000_seeds.json)
and [`bench/results/full_py.json`](bench/results/full_py.json), with per-run
transcripts under [`bench/results/logs/`](bench/results/logs).

| n | instance | warm start | C++ score | Python score | C++ algo s | Python algo s | faster |
|---|---|---|---|---|---|---|---|
| 500 | seed 0 | 3,073 | 3,543 | **3,550** | 0.262 | **0.216** | Python, 1.21x |
| 1,000 | seed 0 | 14,843 | **15,986** | 15,966 | 0.975 | **0.822** | Python, 1.19x |
| 2,000 | seed 0 | 59,731 | **63,993** | 63,907 | 3.747 | **3.601** | Python, 1.04x |
| 4,000 | seed 0 | 247,533 | **265,814** | 265,691 | **23.538** | 25.754 | C++, 1.09x |
| 8,000 | seed s1 | 966,633 | 1,057,633 | **1,057,766** | 134.915 | **123.737** | Python, 1.09x |
| 8,000 | seed s2 | 961,829 | **1,055,586** | 1,055,548 | 168.163 | **166.721** | Python, 1.01x |
| 8,000 | seed 0 — *pathological, see below* | — | 1,062,386 | did not finish | 137.81 | terminated at ~2,580 | not comparable |
| 18,524 | full challenge | 5,154,247 | 5,852,223 | **5,852,589** | 1,841.786 | **1,142.863** | Python, 1.61x |

> **Correction to an earlier version of this README.** On the strength of the
> single seed-0 n=8,000 instance, this file previously reported that `acdc_py`
> hits a scaling "cliff" past n≈4,000 and was more than 18x slower there or did
> not finish at all. **That framing was wrong and is retracted.** Two further
> n=8,000 instances, drawn with different sampling seeds, and the full n=18,524
> challenge instance all complete normally in both implementations — and on all
> three `acdc_py` is the faster of the two, including at full challenge scale.
> The seed-0 stall is real and reproducible, but it is one pathological instance
> rather than a size threshold. See
> [The seed-0 n=8,000 stall](#the-seed-0-n8000-stall).

The seed-0 n=8,000 row appears only in
[`bench/results/sweep.log`](bench/results/sweep.log), not in `sweep.json`:
`bench.py` writes the JSON after each size completes, and that size aborted when
the Python run was terminated, so its warm-start score was never recorded.

### Accuracy: a genuine tie, at every size including full scale

The two implementations are never more than **0.2%** apart anywhere in the
range, the gap narrows as `n` grows, and the sign flips from instance to
instance:

| n | instance | difference | in favour of |
|---|---|---|---|
| 500 | seed 0 | +7 (+0.20%) | Python |
| 1,000 | seed 0 | +20 (+0.13%) | C++ |
| 2,000 | seed 0 | +86 (+0.13%) | C++ |
| 4,000 | seed 0 | +123 (+0.05%) | C++ |
| 8,000 | seed s1 | +133 (+0.013%) | Python |
| 8,000 | seed s2 | +38 (+0.004%) | C++ |
| 18,524 | full challenge | +366 (+0.006%) | Python |

At full challenge scale `acdc_py` scores 5,852,589 against the C++'s 5,852,223 —
Python ahead by 366 points, or 0.006%. Both land within about 0.03% of the
winning leaderboard entry for the challenge (5,853,925: Python is 1,336 short,
the C++ 1,702 short) and both improve on the benchmark warm start of 5,154,247
by 13.5%. On the two fresh n=8,000 instances the difference is a coin flip in
both magnitude and direction.

Neither implementation is meaningfully more accurate than the other. The
residual differences trace to the linear-assignment step described above — tie
breaking between two exact solvers — not to any difference in the search itself.

### Speed: close, and the winner depends on size and instance

There is no clean language win here. Across seven measured instances `acdc_py`
is faster on six of them:

- **Below n≈2,000** Python is clearly ahead, by about 20% at n=500 and n=1,000.
  Numba-compiled inner loops over sparse structure, with the heavy dense work in
  numpy, are enough to erase the language gap at this scale.
- **At n=2,000–8,000** the two are near parity. Python is 4% ahead at n=2,000,
  the C++ is 9% ahead at n=4,000, and on the two fresh n=8,000 instances Python
  is ahead again — by 9% on seed s1 (123.7 s against 134.9 s) and by 1% on seed
  s2 (166.7 s against 168.2 s). Differences of this size are within the range
  that instance-to-instance variation alone can produce.
- **At full challenge scale Python is substantially faster.** `acdc_py`
  completes n=18,524 in 1,142.9 s of algorithm time (19.0 minutes); the C++
  takes 1,841.8 s (30.7 minutes), a factor of 1.61. The C++ profiler puts
  523.6 s in Frank–Wolfe and 1,318.2 s in greedy search, and the `% Total
  elapsed time is 30.7 minutes` line confirms that the algorithm phases are
  essentially the whole run.

The full-scale gap is not in the continuous phase, where the C++ is in fact
faster (523.6 s against 816.9 s). It is in the discrete phase: the C++ greedy
search spends 1,318.2 s against Python's 325.9 s, with 1,056.6 s of that inside
`make_swaps.fast.find_best` and a single one of the five greedy passes taking
554.5 s. Whatever the two greedy implementations do differently, it only becomes
expensive at this size — on n=8,000 seed s1 the C++ greedy phase is 63.8 s
against Python's 33.6 s, the same direction but a much smaller absolute cost.

The C++ does keep a real advantage in end-to-end wall time that `algo_s` hides,
though it shrinks in relative terms as `n` grows: Python pays 0.16–0.25 s of
interpreter and numba import plus numpy CSV parsing before the algorithm starts,
against roughly 1 ms to 48 ms of hand-written parsing for the C++. At full scale
both spend about a second on loading (Python 0.99 s, C++ 1.04 s), which is
noise against a 19- to 31-minute run.

### Memory: a consistent C++ disadvantage

Peak resident set size, from `/usr/bin/time -l`:

| n | instance | C++ peak RSS | Python peak RSS |
|---|---|---|---|
| 500 | seed 0 | 213.5 MB | 168.7 MB |
| 1,000 | seed 0 | 834.0 MB | 339.4 MB |
| 2,000 | seed 0 | 343.9 MB | 319.5 MB |
| 4,000 | seed 0 | 1,344.1 MB | 854.4 MB |
| 8,000 | seed s1 | 5,130.4 MB | 2,958.5 MB |
| 8,000 | seed s2 | 5,131.5 MB | 2,943.7 MB |
| 18,524 | full challenge | not captured | 6,673.5 MB |

At the sizes where memory actually matters this is now a firm finding rather
than the provisional one reported earlier. At n=8,000 the C++ holds about
5.13 GB against Python's 2.95 GB — a 1.74x difference, reproduced to within
2 MB on each side across two independently sampled instances. The n=4,000
reading points the same way at 1.57x. Both implementations hold dense `n x n`
gradient and swap-gain matrices, so both are `O(n^2)`; the difference is in the
constant, and the C++'s is larger.

The small-`n` readings remain noisy and should not be quoted — the C++ measuring
834 MB at n=1,000 and only 344 MB at n=2,000 on the same instance family is not
a real size effect. That noise is immaterial at 5 GB.

The full-scale C++ run was launched without an `/usr/bin/time -l` capture, so
its peak RSS is genuinely unknown and no figure is given for it. Python's
6,673.5 MB at n=18,524 is measured. Filling in the C++ side needs a re-run.

### The seed-0 n=8,000 stall

On one specific instance — the n=8,000 subset sampled with `--seed 0` —
`acdc_py` does not finish. The C++ solves it in 138 s; `acdc_py` had not
returned after 43 minutes and was terminated (the `command failed (-15)` line at
the end of `sweep.log`). This is a real pathology and it reproduces on that
instance.

It is **not** a scaling cliff, and the earlier claim that it was one has been
withdrawn. Two n=8,000 instances built with different sampling seeds
([`bench/results/n8000_seeds.json`](bench/results/n8000_seeds.json)) both
complete in `acdc_py`, in 123.7 s and 166.7 s, in both cases slightly faster
than the C++ on the same instance. The full n=18,524 instance — more than twice
the size — completes in 19 minutes. Size is not what triggers it.

The cause is **not** the bulk numerics. Sampling the stalled process put it
inside SciPy's `min_weight_full_bipartite_matching` (`_lapjvsp`), which
`acdc_py` reaches through `permutation_match(..., solver="sparse")` when it
projects the Frank–Wolfe iterate back onto a permutation. Frank–Wolfe calls
`permutation_match` for two different purposes — the dense solver picks the
search direction from the gradient, the sparse one performs this projection — so
`harness/diag_projection.py` wraps the function and prints every call as it
returns, which separates them. On the seed-0 instance
([`bench/results/diag_n8000.log`](bench/results/diag_n8000.log)) the trace ends
mid-call and identifies the culprit exactly:

- Calls 1–10, the dense direction-finding solves of the first alternation:
  0.96–1.37 s each, as expected at this size.
- Call 11, the **first** sparse projection (nnz 13,055, 620 distinct values):
  0.001 s.
- Calls 12–21, the dense solves of the second alternation: 0.92–1.40 s each.
- Call 22, the **second** sparse projection (nnz 11,181, 771 distinct values):
  never returns. The log stops on its opening line.

So the stall is a single call to SciPy's sparse matcher, on the second
alternation, on an input that is no larger or denser than the one it handled in
a millisecond on the first. Everything around it is healthy. The same projection
costs 0.79 ms in total at n=4,000 and 1.39 ms at n=8,000 in the per-scope
profiles ([`bench/results/prof_py_n*.json`](bench/results)); on the healthy
n=8,000 seeds and at n=18,524 it is equally cheap — the full-scale Python log
shows its sparse projection completing in 0.002 s at nnz 25,183. The seed-0
instance with `outer=1` also completes, in 29.6 s, because it stops before
reaching call 22.

The reproducer has been isolated: `diag_projection.py --dump-dir` saves the
input matrix of each sparse projection, so the exact matrix that hangs is
available without re-running the search. What is established is that this is an
input-pattern pathology in a specific SciPy version's sparse Jonker–Volgenant
code, reachable from `acdc_py` but not a property of `acdc_py`'s own numerics,
and rare — one instance in three at n=8,000, absent at every other size tested.
What is still open is which property of that matrix triggers it. The dumped
matrices are derived from the challenge data and so are not committed here; see
[Licensing and data](#licensing-and-data).

### Phase split: the same algorithm, diverging balance

Share of algorithm time spent in Frank–Wolfe (the remainder is greedy search):

| n | instance | C++ | Python |
|---|---|---|---|
| 500 | seed 0 | 78% | 80% |
| 1,000 | seed 0 | 77% | 74% |
| 2,000 | seed 0 | 72% | 80% |
| 4,000 | seed 0 | 62% | 60% |
| 8,000 | seed s1 | 53% | 73% |
| 8,000 | seed s2 | 52% | 65% |
| 18,524 | full challenge | 28% | 71% |

Up to n=4,000 the two agree to within 3 points at three of the four sizes
(8 points at n=2,000), and the Frank–Wolfe share declines in both as the greedy
swap phase grows faster than the continuous one. Combined with the
near-identical scores, that remains good evidence that both are executing the
same algorithm rather than merely arriving at similar answers.

Beyond n=4,000 the balance comes apart. Python's Frank–Wolfe share stops falling
and settles around 65–73%, while the C++'s keeps dropping to 28% at full scale,
because its greedy phase grows much faster than Python's. This is the same
observation as the full-scale speed result seen from the other side: the two
implementations still agree on the answer, and on which phases exist, but they
do not agree on the cost of the discrete phase at challenge scale.

### Full challenge scale (n=18,524)

Both implementations complete the untouched challenge instance (4,158,056 male
edges, 2,208,679 female edges), warm-started from the benchmark matching, with
the same `outer=5`, `fw=10` parameters as every other size.

| | score | vs winner | algo s | Frank–Wolfe s | greedy s | load s | peak RSS |
|---|---|---|---|---|---|---|---|
| `vnc-translation` (C++) | 5,852,223 | −1,702 (−0.029%) | 1,841.786 | 523.611 | 1,318.175 | 1.036 | not captured |
| `acdc_py` (Python) | **5,852,589** | −1,336 (−0.023%) | **1,142.863** | 816.928 | **325.932** | 0.987 | 6,673.5 MB |

Both scores were re-derived with `harness/official_score.py` against the
challenge CSVs. Both are within 0.03% of the winning leaderboard entry
(5,853,925) and 13.5% above the benchmark warm start (5,154,247), and they are
within 0.006% of each other, with Python ahead.

The headline is that full scale does not rescue the language advantage — it
reverses it. `acdc_py` is 1.61x faster in algorithm time here, entirely because
of the greedy phase, and it does the job in 6.7 GB. The dense `n x n` working
matrices alone are about 2.7 GB each at this size, so both are dominated
by memory traffic rather than by arithmetic, which is the most likely reason the
ranking at full scale differs from the ranking at n=4,000.

### Caveats

The subset instances are random induced subgraphs of the challenge connectomes:
`prep.py` samples `n` matched node pairs and keeps the edges among them. Two
instances of the same size are therefore different problems, and per-instance
variance in both runtime and score is real, not measurement error. The n=8,000
seed sensitivity is the sharpest illustration — one seed produces a hang in
`acdc_py` and two produce ordinary 2-minute runs — but it applies to every row
in the tables above.

Concretely: sizes n=500 through n=4,000 and n=18,524 rest on a single instance
each, and the speed differences at n=2,000–8,000 are in the few-percent range.
Those margins should not be read as settled. Establishing the crossover
behaviour with confidence would need several seeds per size, reported with a
spread rather than a single number. The claims that do survive that objection
are the ones that hold across every instance measured: accuracy is a tie, memory
favours Python at the larger sizes, and the full-scale speed gap of 1.61x is far
too large to be seed noise.

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

Add `--full` to include the untouched n=18,524 challenge instance. Expect about
half an hour per implementation at that size, and note that `bench.py` caches
subset instances as `instances/n{n}` keyed on size alone, so a differently
seeded instance of the same size needs its own directory:

```bash
cd bench
./.venv/bin/python harness/prep.py \
    --src data --dst instances/n8000s1 --n 8000 --seed 1
./.venv/bin/python harness/prep.py \
    --src data --dst instances/n8000s2 --n 8000 --seed 2
```

Those are the two instances behind the `n8000s1` and `n8000s2` rows; each was
then run through the same `bench_size` path as every other row, with
`--repeats 2` and the same `outer=5`, `fw=10`.

To reproduce the per-scope Python profile of a healthy run:

```bash
cd bench
./.venv/bin/python harness/profile_py.py \
    --csv-dir instances/n8000 --outer 1 --fw 10 \
    --json results/prof_py_n8000.json
```

To reproduce the trace that located the seed-0 stall — `--dump-dir` writes the
input matrix of each sparse projection, so the hanging one is preserved even
though the process has to be killed:

```bash
cd bench
./.venv/bin/python harness/diag_projection.py \
    --csv-dir instances/n8000 --outer 5 --fw 10 \
    --dump-dir results/dumps | tee results/diag_n8000.log
```

Individual pieces are usable on their own: `harness/prep.py` builds a subset
instance, `harness/run_py.py` is a standalone `acdc_py` driver with
`main_acdc`'s interface, and `harness/official_score.py` scores any
`(male, female, matching)` CSV triple — which is how the two full-scale scores
above were verified:

```bash
cd bench/harness
../.venv/bin/python official_score.py \
    --male ../data/male_connectome_graph.csv \
    --female ../data/female_connectome_graph.csv \
    --matching <solution.csv>
```

## Environment the numbers were taken on

- Apple M4 Max, 16 cores (12 performance), 64 GB RAM, macOS 15.7.7 (24G720)
- Python 3.14.7, numpy 2.5.3, scipy 1.18.1, numba 0.67.0
- Apple clang 17.0.0 (arm64), CMake Release build, Eigen and Homebrew libomp

Absolute timings are specific to this machine. The cross-implementation ratios
should carry over more readily than the seconds do, but the seed-0 n=8,000 stall
in particular depends on a specific SciPy version's sparse assignment code and
is worth re-checking against whatever SciPy a reader has.

## Repository contents

```
bench/harness/          the benchmark code
  prep.py               build reduced instances from the challenge CSVs
  run_py.py             acdc_py driver with main_acdc's command line
  official_score.py     the challenge's own scoring formula, applied to files
  bench.py              run both implementations and compare
  profile_py.py         per-scope attribution of acdc_py's runtime
  diag_projection.py    trace every permutation_match call to find the stall
bench/results/          measurements (timings and scores only, no graph data)
  sweep.json            n=500-4,000, in full detail
  sweep.log             console transcript of the sweep
  n8000_seeds.json      the two extra n=8,000 instances (seeds 1 and 2)
  full_py.json          acdc_py at full challenge scale
  results.json          an earlier partial pass (n=500, n=1,000)
  prof_py_n*.json       per-scope Python profiles at n=2,000/4,000/8,000
  diag_n8000.log        the permutation_match trace that isolates the stall
  logs/                 per-run stdout and /usr/bin/time -l captures
scripts/fetch_data.sh   download the challenge inputs
```

Not in the repository, and gitignored: `bench/data/`, `bench/instances/` and
`bench/work/` (challenge data and everything derived from it),
`bench/results/dumps/` (the saved sparse projection matrices from the stall
diagnosis, which are also derived from the challenge data), `bench/.venv/`, and
the two upstream clones.

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
