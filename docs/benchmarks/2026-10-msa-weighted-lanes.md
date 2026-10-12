# Weighted lane timings

The default stays `MsaLane::I32`. F32 is slower at both read counts and does not pass the decision rule.
Both lanes remain available.

This benchmark compares the unit and weighted lanes for multi-spin coded simulated annealing, or `MSA`.
It also measures weighted simulated annealing, or `SA`, on the Advantage2 mining topology.
The results answer `QUI-1641` and measure the cost of weights on the 64-lane kernel for `QUI-1398`.

## Setup

The run uses device 0, an `NVIDIA RTX A4000` with compute capability 8.6 and a 140 W power limit.
The grid runs on October 11 and 12, 2026, in `US/Eastern` time.
The driver is 610.57.04. The `quip-solver-cuda-ci` image supplies `CUDA 12.9.1` and `NVCC 12.9.86`.
The run starts from commit `e729148`. It adds the weighting option and energy metrics in this change.
The kernels match that base commit.

The topology is `/zfspool/quip-mining/isingmark/fixtures/advantage2-system1.spec.json`.
It has 4,577 nodes, 41,515 edges, and a degree limit of 20.
The `RTX A4000` holds this graph and serves 128 `MSA` reads. The run uses the real topology throughout.
The topology file has `SHA-256`:

```text
f72af98b2bd6c1217d6d3c6389b3ba3e2790274d9e97822945c9f65b701ff4d3
```

The eight corpus nonces come from the fixed seed `qui1641-task11-advantage2-v1`.
For each index `i` from 0 through 7, hash the `ASCII` string `qui1641-task11-advantage2-v1:<i>`.
The nonce is the `SHA-256` hex digest.
The corpus file is `/tmp/qui1641-task11/corpus.jsonl`. Its `SHA-256` is:

```text
c0e88ba79af73075765a4ae60b861218c37ebfbc9c3696ef44439d0a19fad7e0
```

The nonces, in index order, are:

```text
6bd61f87d168143eec4b6fc4bdd911c975e9237e2c7f598b79e0101838e9f3b6
88cdbd8a8a9ea296e4fc57c480158281de41bc24838d73dd0885c1aec03b54eb
f25c2000512149d597af567233313870c10b59c2e7e9758c9ee8367f6e5ac50f
e9fae0f3660334d09dd70d8ee21f1b7162b168acd0d989450514e002da38ce20
567eb89d7f4148ef25262cb310be69fa5b5048f5f5d3782ede51fc6fcecdbc99
6b0abda9e51cb747424b36652a49ddce1371868b0edff28f746c3eb2f56e34f4
98116f03b7a26ab70076e449a19d8026b4aba43ce7a66f4916c71cabdbaf86d1
02fa84af47b6e24a70cb68c6a1b7ac66a07d68455b3c311fd230236ddc08d4d5
```

`--weights 1`, the default, preserves the corpus redraw: `J` is ±1 and `h` is −1, 0, or 1.
`--weights 5` seeds `ChaCha8` with each model nonce.
It draws fields first, then coupling magnitudes, in graph order.
Each field is in −5 through 5. Each coupling keeps its sign and has an absolute value in 1 through 5.
The draw uses the protocol random number generator and modulo rule. All lanes receive the same models.
A synthetic model without a 64-hex nonce uses the all-zero seed.
Sampling keeps `SampleParams::default().seed = 0` for each call, including repeats.

## Method

Each cell uses eight models. Each model has one warm-up and five measured repeats.
The grid covers 64 and 128 reads at 256, 1,024, and 4,096 sweeps, with four sweeps per beta rung.
Each row has 40 measured line-delimited JSON records. Warm-ups do not enter the results.
The run tests one cell at a time on device 0 under taskd.

For each repeat index, average the eight models' `sweep.per_call_ns` values.
The timing table reports the median of those five averages. The spread is the largest value minus the smallest value.
These times come from `CUDA` events. Each value divides one model's kernel time by its sweep count.
They exclude compilation, uploads, and host scoring.
The spread shows timing changes for this fixed corpus. It does not give a confidence interval.

Each record gives the lowest read energy as `best_energy`.
For each model, average those five repeat values. The mean energy averages the eight model means.
The lowest energy is the lowest value observed across all models and repeats.
The values use energy units. Divide milli units by 1,000 to get these units.
Lower energy is better. The host scores the returned spins with the consensus scorer.
Repeats keep the sampling seed fixed and do not represent five independent random seeds.

Each row has a target from weighted `SA`. The target is its mean best-read energy at the same read count and sweep budget.
Search the path's three sweep budgets. Choose the first budget whose mean best-read energy reaches the target.
At that budget, average the whole-model wall times across the corpus for each repeat. The table gives the median of those five averages.
It includes per-call setup, transfer, polling, and scoring.
Compilation, corpus loading, and weighting occur outside this clock.
This time describes the first measured sweep budget that reaches the target. The benchmark records energies at the end of each anneal.
The table uses those measured points without interpolation.
The 4,096-sweep rows use the largest tested `SA` budget as the target.
The unit rows use different weights from `SA`. The energy values of those rows cannot measure how well weighted `SA` solves the model.

The line-delimited JSON records supply the results. `bench fold` combines external Nsight reports and does not combine repeats.
Raw records, run logs, and the summary script are under `/tmp/qui1641-task11/`.

## Reproduction

Build the binaries with the same image and absolute worktree mount:

```sh
docker run --rm -e CI=1 \
  -v /zfspool/quip-mining/quip-solver-cuda-wt-qui-1641:/src \
  -v quip-cuda-cargo:/root/.cargo/registry \
  -v quip-cuda-target:/src/target \
  -w /src quip-solver-cuda-ci cargo build --release --bins
```

Run each cell in this container:

```sh
docker run --rm --gpus all -e CI=1 \
  -v /zfspool/quip-mining/quip-solver-cuda-wt-qui-1641:/src \
  -v quip-cuda-cargo:/root/.cargo/registry \
  -v quip-cuda-target:/src/target \
  -v /tmp/qui1641-task11:/tmp/qui1641-task11 \
  -v /zfspool/quip-mining/isingmark/fixtures/advantage2-system1.spec.json:/zfspool/quip-mining/isingmark/fixtures/advantage2-system1.spec.json:ro \
  -w /src quip-solver-cuda-ci \
  /src/target/release/quip-cuda-msa --device 0 bench run \
  --source /tmp/qui1641-task11/corpus.jsonl \
  --topology /zfspool/quip-mining/isingmark/fixtures/advantage2-system1.spec.json \
  --reads 64 --sweeps 256 --sweeps 1024 --sweeps 4096 \
  --sweeps-per-beta 4 --warmup 1 --repeats 5 \
  --msa-path auto --out /tmp/qui1641-task11/grid/unit-auto-r64
```

Repeat at `--reads 128`. Use a separate output directory for each row in this path table:

| Path | Binary | Extra arguments |
| --- | --- | --- |
| Unit auto | `quip-cuda-msa` | `--msa-path auto` |
| Unit I32 | `quip-cuda-msa` | `--msa-path weighted-i32` |
| Unit F32 | `quip-cuda-msa` | `--msa-path weighted-f32` |
| Weighted I32 | `quip-cuda-msa` | `--msa-path weighted-i32 --weights 5` |
| Weighted F32 | `quip-cuda-msa` | `--msa-path weighted-f32 --weights 5` |
| Weighted SA | `quip-cuda-sa` | `--weights 5` |

## Results at 64 reads

Timing values are microseconds per sweep. The target-time column gives milliseconds and the first sweep budget that reaches the target.

| Path | Sweeps | Median µs | Spread µs | Mean energy | Lowest energy | Time to target |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Unit auto | 256 | 109.528 | 3.054 | −14652.375 | −14700 | not reached |
| Unit auto | 1,024 | 110.086 | 0.032 | −14736.625 | −14772 | not reached |
| Unit auto | 4,096 | 107.679 | 0.074 | −14779.625 | −14822 | not reached |
| Unit I32 | 256 | 351.607 | 0.487 | −14658.625 | −14702 | not reached |
| Unit I32 | 1,024 | 352.032 | 1.093 | −14738.375 | −14780 | not reached |
| Unit I32 | 4,096 | 351.809 | 0.054 | −14784.125 | −14844 | not reached |
| Unit F32 | 256 | 389.794 | 0.534 | −14649.875 | −14685 | not reached |
| Unit F32 | 1,024 | 390.308 | 0.661 | −14741.125 | −14780 | not reached |
| Unit F32 | 4,096 | 389.869 | 0.486 | −14779.625 | −14830 | not reached |
| Weighted I32 | 256 | 355.552 | 0.167 | −48654.125 | −48931 | 116.668 ms at 256 |
| Weighted I32 | 1,024 | 355.377 | 1.276 | −48982.875 | −49245 | 1478.021 ms at 4,096 |
| Weighted I32 | 4,096 | 354.223 | 0.045 | −49115.125 | −49334 | not reached |
| Weighted F32 | 256 | 391.476 | 0.494 | −48648.875 | −48888 | 126.149 ms at 256 |
| Weighted F32 | 1,024 | 389.555 | 0.295 | −48988.625 | −49262 | 422.220 ms at 1,024 |
| Weighted F32 | 4,096 | 388.717 | 0.428 | −49127.375 | −49321 | 1613.971 ms at 4,096 |
| Weighted `SA` | 256 | 10717.782 | 5.524 | −48631.625 | −48936 | 2769.936 ms at 256 |
| Weighted `SA` | 1,024 | 10463.598 | 0.678 | −48987.375 | −49242 | 10739.928 ms at 1,024 |
| Weighted `SA` | 4,096 | 10236.862 | 1.131 | −49125.375 | −49358 | 41957.509 ms at 4,096 |

## Results at 128 reads

Timing values are microseconds per sweep. The target-time column gives milliseconds and the first sweep budget that reaches the target.

| Path | Sweeps | Median µs | Spread µs | Mean energy | Lowest energy | Time to target |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Unit auto | 256 | 98.077 | 0.039 | −14666.125 | −14714 | not reached |
| Unit auto | 1,024 | 97.611 | 0.018 | −14751.125 | −14802 | not reached |
| Unit auto | 4,096 | 97.157 | 0.012 | −14787.375 | −14842 | not reached |
| Unit I32 | 256 | 551.827 | 0.230 | −14667.125 | −14692 | not reached |
| Unit I32 | 1,024 | 550.847 | 0.268 | −14747.125 | −14796 | not reached |
| Unit I32 | 4,096 | 550.791 | 0.311 | −14784.375 | −14830 | not reached |
| Unit F32 | 256 | 632.610 | 0.352 | −14655.625 | −14700 | not reached |
| Unit F32 | 1,024 | 631.568 | 0.071 | −14751.375 | −14788 | not reached |
| Unit F32 | 4,096 | 631.283 | 0.134 | −14785.625 | −14828 | not reached |
| Weighted I32 | 256 | 551.788 | 0.159 | −48707.375 | −48921 | 176.279 ms at 256 |
| Weighted I32 | 1,024 | 550.851 | 0.265 | −48994.875 | −49217 | 2300.526 ms at 4,096 |
| Weighted I32 | 4,096 | 550.811 | 0.305 | −49151.375 | −49350 | 2300.526 ms at 4,096 |
| Weighted F32 | 256 | 631.002 | 0.307 | −48663.375 | −48905 | 196.882 ms at 256 |
| Weighted F32 | 1,024 | 629.745 | 0.102 | −48993.875 | −49197 | 2621.819 ms at 4,096 |
| Weighted F32 | 4,096 | 629.141 | 0.194 | −49140.625 | −49384 | 2621.819 ms at 4,096 |
| Weighted `SA` | 256 | 11978.976 | 9.561 | −48656.125 | −48936 | 3111.404 ms at 256 |
| Weighted `SA` | 1,024 | 11604.210 | 7.654 | −49003.375 | −49246 | 11925.882 ms at 1,024 |
| Weighted `SA` | 4,096 | 11349.613 | 11.407 | −49131.125 | −49358 | 46535.598 ms at 4,096 |

## Cost of weights on the 64-lane kernel

The unit rows use the same ±1 models. Their time ratios isolate the cost of forcing weighted arithmetic on those models.
A ratio greater than 1 means the weighted lane takes more time per sweep than the unit kernel.
At 128 reads, each spin has two 64-lane words.
The last two columns compare the weighted paths on the same K = 5 models. They give the per-sweep speed gain over weighted `SA`.

| Reads | Sweeps | Unit I32 / unit auto | Unit F32 / unit auto | Weighted `SA` / I32 | Weighted `SA` / F32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 256 | 3.210× | 3.559× | 30.144× | 27.378× |
| 64 | 1,024 | 3.198× | 3.545× | 29.444× | 26.860× |
| 64 | 4,096 | 3.267× | 3.621× | 28.899× | 26.335× |
| 128 | 256 | 5.626× | 6.450× | 21.709× | 18.984× |
| 128 | 1,024 | 5.643× | 6.470× | 21.066× | 18.427× |
| 128 | 4,096 | 5.669× | 6.498× | 20.605× | 18.040× |

## Lane decision

Keep `DEFAULT_MSA_LANE = MsaLane::I32` unless F32 is faster per sweep by more than the larger repeat spread at both read counts.
Its best energies must also be no worse.
The table checks that rule on the weighted rows at each sweep budget. The energy check covers both mean and lowest corpus energy.
The time margin is the I32 median minus the F32 median. A positive margin favors F32.

| Reads | Sweeps | Time margin µs | Larger spread µs | Time clears spread | F32 energy no worse |
| ---: | ---: | ---: | ---: | --- | --- |
| 64 | 256 | −35.924 | 0.494 | no | no |
| 64 | 1,024 | −34.179 | 1.276 | no | yes |
| 64 | 4,096 | −34.494 | 0.428 | no | no |
| 128 | 256 | −79.215 | 0.307 | no | no |
| 128 | 1,024 | −78.893 | 0.265 | no | no |
| 128 | 4,096 | −78.330 | 0.305 | no | no |

F32 does not win by the rule. The default stays `MsaLane::I32`. Both lanes remain available.

## Limits

These results cover eight fixed models on one GPU model. They measure isolated model launches.
Production throughput also depends on concurrent models and persistent sessions.
Host validation overlaps part of the early 64-read grid. Whole-model wall times include host work during that period.
The three sweep budgets give endpoint energies. They do not show when a target was first reached within an anneal.
Energy changed across repeats in 0 of the 288 model-and-budget groups.
The lane choice follows the preceding timing and energy rule. Broader device tests can inform a later choice.

The wrapper returned exit 127 after the last benchmark completed. Formatting the running script caused a trailing shell error.
All 12 benchmark invocations completed. The analyzer verified all 36 cells and 1,440 records.
