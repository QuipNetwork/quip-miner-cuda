# Spill measurements on the A4000

The MSA kernel keeps all spins in shared memory up to N = 5,791 nodes at 128 reads.
Past that size, throughput falls in two steps.
At degree 20 in natural index order, one model drops 30% between 5,791 and 8,000 nodes, and 48 models drop 40%.
At 48 models the second step comes near 11,600 nodes, as the L2 estimate predicts, but only when the node labels are in shuffled order.
At 200,000 nodes and 4,000,000 edges (degree 40), 48 models run at 22.6 G lane-updates/s.
That is 21% of the all-shared degree-40 peak of 106.5 G lane-updates/s.
The hypothesis that DRAM traffic at the ceiling comes mainly from topology is not supported (see Breakdown).

## Setup

| Item | Value |
| --- | --- |
| Card | NVIDIA RTX A4000, physical GPU 1, compute capability 8.6 |
| Facts used | 48 SMs, 99 KB opt-in shared memory per block, 4 MB L2, 448 GB/s GDDR6 |
| Driver | 610.57.04, power limit 140 W |
| Image | `quip-solver-cuda-ci`, `sha256:294450ef352cdc40a04dff7ccc1f7f1eaffc9cf10f6d7b6e61206edbd1e9573c` |
| Commit | `0cdfa73` (branch `feat/msa-dram-spill`), binary built from this commit |
| Run | October 21, 2026, 16:34 to 17:16 Eastern time |
| Idle check | GPU 1 showed 0% load and 9 MiB used before the `ncu` slice and after the grid |

GPU 0 took no part in the runs.
The idle check for the grid start ran in the same step as a blocked command, so no reading exists from the instant before the first cell.
The first sampler reading, at the start of the first cell, showed 177 MiB used on GPU 1. Another process using the card is unlikely but not excluded.
The full `nvidia-smi -q` output is in `device.txt` under the evidence path.

## Method

Every cell runs a synthetic circulant graph with all fields `h = 0` and all couplings `J = 1`.
The graph has degree d, in natural index order (plain) or with shuffled node labels (`--shuffle`).
Each cell uses 128 reads, 4 sweeps per beta rung, 1 warm-up, and 3 repeats.
Each cell runs once at 64 sweeps and once at 256 sweeps.

The per-sweep cost is the difference between the launch time at 256 sweeps and at 64 sweeps, divided by 192.
This cancels initialization, packing, and launch overhead.
Each table reports the median over the three repeats.
"G lane-updates/s" is N × 128 × models ÷ per-sweep nanoseconds.
A lane-update is one spin update in one read of one model.

`--nonces` sets the number of models annealed at once. Each model runs on its own SM, so 48 models fill the A4000.
`--msa-shared-spins K` caps the spins held in shared memory. Without it, K is the smaller of N and 5,791.
At 128 reads each spin has two 64-lane words, so the shared budget holds 5,791 spins.
Spins past K live in a per-model global buffer of (N − K) × 16 bytes.

The grid has 109 cells and every one completed:

- Section A: 96 cells. Each combination of degree (20 or 40), order, and model count (1 or 48) runs 12 node counts from 1,000 to 200,000.
- Section B: 10 cells. N = 5,791 and degree 20 with five values of K from 5,791 down to 0, at both model counts.
- Section C: 3 cells. The weighted `i32` lane (`--msa-path weighted-i32`) at degree 40 and 48 models, N = 5,791, 64,000, and 200,000.

The `ncu` slice has 8 kernel profiles at degree 40, plain order, 16 sweeps, one repeat, with N = 5,791, 16,000, 64,000, and 200,000 at 1 and 48 models.

## Results

### Node sweep

Each table pairs one degree and one order. Time is the per-sweep cost in microseconds. Throughput is in G lane-updates/s.

### Degree 20, plain order

| N | shared K | 1 model µs | 1 model G/s | 48 models µs | 48 models G/s |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 1,000 | 48.1 | 2.66 | 48.5 | 126.61 |
| 2,000 | 2,000 | 91.8 | 2.79 | 93.0 | 132.14 |
| 4,000 | 4,000 | 128.7 | 3.98 | 141.9 | 173.15 |
| 5,791 | 5,791 | 123.5 | 6.00 | 180.7 | 196.87 |
| 8,000 | 5,791 | 244.2 | 4.19 | 416.6 | 117.99 |
| 11,582 | 5,791 | 396.5 | 3.74 | 737.0 | 96.55 |
| 16,000 | 5,791 | 538.9 | 3.80 | 1,078.5 | 91.15 |
| 32,000 | 5,791 | 1,172.0 | 3.49 | 2,470.1 | 79.59 |
| 64,000 | 5,791 | 3,029.1 | 2.70 | 5,992.3 | 65.62 |
| 100,000 | 5,791 | 5,066.8 | 2.53 | 10,008.8 | 61.39 |
| 150,000 | 5,791 | 7,599.1 | 2.53 | 16,136.4 | 57.11 |
| 200,000 | 5,791 | 10,160.2 | 2.52 | 22,095.4 | 55.61 |

### Degree 20, shuffled

| N | shared K | 1 model µs | 1 model G/s | 48 models µs | 48 models G/s |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 1,000 | 42.1 | 3.04 | 42.4 | 144.80 |
| 2,000 | 2,000 | 80.1 | 3.20 | 80.4 | 152.79 |
| 4,000 | 4,000 | 117.5 | 4.36 | 127.6 | 192.60 |
| 5,791 | 5,791 | 131.1 | 5.66 | 148.3 | 239.94 |
| 8,000 | 5,791 | 269.7 | 3.80 | 377.6 | 130.18 |
| 11,582 | 5,791 | 410.4 | 3.61 | 1,110.4 | 64.09 |
| 16,000 | 5,791 | 553.4 | 3.70 | 1,915.3 | 51.33 |
| 32,000 | 5,791 | 1,257.3 | 3.26 | 5,749.1 | 34.20 |
| 64,000 | 5,791 | 3,207.3 | 2.55 | 15,464.6 | 25.43 |
| 100,000 | 5,791 | 5,120.6 | 2.50 | 26,846.4 | 22.89 |
| 150,000 | 5,791 | 8,217.0 | 2.34 | 42,577.4 | 21.65 |
| 200,000 | 5,791 | 13,672.8 | 1.87 | 58,331.6 | 21.07 |

### Degree 40, plain order

| N | shared K | 1 model µs | 1 model G/s | 48 models µs | 48 models G/s |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 1,000 | 96.9 | 1.32 | 97.3 | 63.12 |
| 2,000 | 2,000 | 124.0 | 2.06 | 129.9 | 94.58 |
| 4,000 | 4,000 | 238.1 | 2.15 | 241.4 | 101.82 |
| 5,791 | 5,791 | 276.9 | 2.68 | 334.1 | 106.50 |
| 8,000 | 5,791 | 513.0 | 2.00 | 844.8 | 58.18 |
| 11,582 | 5,791 | 790.7 | 1.88 | 1,593.7 | 44.65 |
| 16,000 | 5,791 | 1,100.4 | 1.86 | 2,540.8 | 38.69 |
| 32,000 | 5,791 | 2,641.9 | 1.55 | 6,739.5 | 29.17 |
| 64,000 | 5,791 | 5,275.5 | 1.55 | 15,900.7 | 24.73 |
| 100,000 | 5,791 | 10,383.9 | 1.23 | 25,847.9 | 23.77 |
| 150,000 | 5,791 | 16,132.3 | 1.19 | 40,097.0 | 22.98 |
| 200,000 | 5,791 | 21,607.8 | 1.18 | 54,407.6 | 22.59 |

### Degree 40, shuffled

| N | shared K | 1 model µs | 1 model G/s | 48 models µs | 48 models G/s |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 1,000 | 90.0 | 1.42 | 90.6 | 67.82 |
| 2,000 | 2,000 | 135.9 | 1.88 | 117.7 | 104.40 |
| 4,000 | 4,000 | 238.8 | 2.14 | 271.7 | 90.44 |
| 5,791 | 5,791 | 277.2 | 2.67 | 322.9 | 110.18 |
| 8,000 | 5,791 | 563.8 | 1.82 | 901.6 | 54.52 |
| 11,582 | 5,791 | 835.2 | 1.78 | 2,153.6 | 33.04 |
| 16,000 | 5,791 | 1,250.8 | 1.64 | 3,861.3 | 25.46 |
| 32,000 | 5,791 | 2,874.3 | 1.43 | 11,254.9 | 17.47 |
| 64,000 | 5,791 | 5,888.5 | 1.39 | 30,273.4 | 12.99 |
| 100,000 | 5,791 | 9,313.2 | 1.37 | 52,456.4 | 11.71 |
| 150,000 | 5,791 | 14,276.5 | 1.34 | 83,664.4 | 11.02 |
| 200,000 | 5,791 | 25,500.9 | 1.00 | 115,034.0 | 10.68 |

The all-shared peak is at N = 5,791 in every series.
Past 5,791, one model at degree 20 falls from 6.00 to 4.19 G/s at 8,000 nodes and then declines to 2.52 G/s at 200,000.
The curve is flat from 100,000 to 200,000 nodes in plain order.
Forty-eight models fall from 196.9 to 118.0 G/s over the same step, and to 55.6 G/s at 200,000.

Shuffled labels cost nothing while every spin is in shared memory.
At N = 5,791 the shuffled series are within 6% of the plain series at one model and faster at 48 models.
Once spins spill, the shuffled series fall much further, most of all at 48 models.

### Shared fraction at constant work

N = 5,791 and degree 20, plain order. Only the placement changes.

| Models | Shared K | Sweep (µs) | Versus all-shared |
| ---: | ---: | ---: | ---: |
| 1 | 5,791 | 160.7 | 1.00x |
| 1 | 4,000 | 207.6 | 1.29x |
| 1 | 2,000 | 226.7 | 1.41x |
| 1 | 1,000 | 199.9 | 1.24x |
| 1 | 0 | 211.9 | 1.32x |
| 48 | 5,791 | 164.1 | 1.00x |
| 48 | 4,000 | 332.8 | 2.03x |
| 48 | 2,000 | 408.3 | 2.49x |
| 48 | 1,000 | 352.6 | 2.15x |
| 48 | 0 | 381.3 | 2.32x |

Moving the last 1,791 spins out of shared memory costs 29% for one model and 103% for 48 models.
Moving more spins out adds little.
The cost is not proportional to the spilled fraction, and it does not rise monotonically as K falls.
The same configuration (N = 5,791, degree 20, K = 5,791, one model) measured 123.5 µs in the node sweep and 160.7 µs here.
Single-model cells carry a spread of about 30% between cells.
The 48-model cells agree within 10% (180.7 and 164.1 µs).
The K sweep at one model is too noisy to rank K = 4,000 against K = 0.

### Weighted i32 lane, degree 40, 48 models

| N | Weighted sweep (µs) | Unit d40 plain sweep (µs) | Weighted / unit | Weighted G/s |
| ---: | ---: | ---: | ---: | ---: |
| 5,791 | 2,900.3 | 334.1 | 8.68x | 12.27 |
| 64,000 | 104,537.7 | 15,900.7 | 6.57x | 3.76 |
| 200,000 | 343,328.0 | 54,407.6 | 6.31x | 3.58 |

The weighted `i32` lane costs 6.3 to 8.7 times the unit lane at degree 40.
In the Task 4b design, a spin of degree over 20 on the weighted lanes reloads its neighbour words once per 8-lane group.
Degree 40 exceeds that threshold for every spin, so this section measures that reload.
The 8.7x cost at N = 5,791 occurs with no spill, so the reload and the wider coefficients cause it, not placement.
No weighted run at degree 20 exists in this grid, so the reload cost is not separated from the cost of 4-byte coefficients.
A weighted degree-40 model at 200,000 nodes takes 91.8 s end to end, against 17.4 s for the unit lane.

### Host parts at 200,000 nodes

Milliseconds, median over three repeats, at 256 sweeps.

| Series | Setup | Upload | Download | Energy score | Model total |
| --- | ---: | ---: | ---: | ---: | ---: |
| d20 plain, 1 model | 5 | 7 | 25 | 1,372 | 4,132 |
| d20 plain, 48 models | 8 | 315 | 17 | 1,362 | 7,477 |
| d20 shuffled, 1 model | 4 | 10 | 21 | 1,379 | 5,189 |
| d20 shuffled, 48 models | 8 | 467 | 16 | 1,338 | 17,068 |
| d40 plain, 1 model | 15 | 13 | 32 | 2,625 | 8,532 |
| d40 plain, 48 models | 12 | 611 | 24 | 2,600 | 17,421 |
| d40 shuffled, 1 model | 8 | 21 | 32 | 2,656 | 9,815 |
| d40 shuffled, 48 models | 13 | 1,062 | 28 | 2,637 | 33,749 |

At the ceiling (d40 plain, 48 models) the kernel takes 13.8 s of the 17.4 s total.
Host scoring takes 2.6 s and upload takes 0.6 s.
Problem setup and download together take under 40 ms.
The energy score time is the same at one model and at 48 models, which indicates that each record scores the reads of one model.
It grows with edge count and does not depend on order.
In shuffled order the kernel dominates even more: 33.7 s in total for degree 40 at 48 models.

### Cache profile

Degree 40, plain order, 16 sweeps, one repeat. The profiler serializes the kernel and fixes the clocks.
At N = 200,000 and 48 models the profiled kernel takes 1.246 s, which is 77.9 ms per sweep.
The grid measured 54.4 ms per sweep, so profiled times run about 43% longer.
The hit rates apply to the profiled run.

| Models | N | L1 hit % | L2 hit % | DRAM % of peak | SM % of peak |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5,791 | 75.4 | 96.8 | 0.0 | 0.7 |
| 1 | 16,000 | 52.0 | 94.2 | 0.1 | 0.5 |
| 1 | 64,000 | 50.6 | 85.1 | 0.4 | 0.5 |
| 1 | 200,000 | 50.5 | 66.2 | 0.8 | 0.4 |
| 48 | 5,791 | 77.1 | 93.2 | 4.3 | 31.7 |
| 48 | 16,000 | 52.3 | 78.7 | 17.1 | 18.7 |
| 48 | 64,000 | 51.0 | 64.3 | 24.3 | 13.8 |
| 48 | 200,000 | 50.6 | 59.7 | 25.5 | 12.6 |

With 48 models, the L2 hit rate falls from 93.2% to 59.7% and DRAM use rises from 4.3% to 25.5% of peak.
DRAM use then levels off. At 200,000 nodes about 75% of DRAM bandwidth is idle.
Warps active is 16.7% of the peak at 48 models and 200,000 nodes.
One model leaves DRAM close to idle at every size (below 1%) and still falls to 66% L2 hits at 200,000 nodes.
Both observations point to latency-bound access, not a bandwidth limit.

## Breakdown

**1. Tier map.** Spin state sits in one of three tiers.

| Tier | 1 model | 48 models |
| --- | --- | --- |
| All shared | N up to 5,791 | N up to 5,791 |
| Spilled, within L2 | 5,791 to about 250,000 (estimate) | 5,791 to about 11,000 (estimate) |
| Spilled, to DRAM | not reached by the 200,000 ceiling | beyond about 11,000 |

The first transition shows in throughput between 5,791 and 8,000 nodes in every series.
The drop is 25% to 30% at one model and 40% to 45% at 48 models, in plain order.
The measurement brackets it but does not locate it more finely, because the grid has no node count between 5,791 and 8,000.

The second transition appears at 48 models between 8,000 and 11,582 nodes with shuffled labels.
Degree-20 shuffled throughput halves from 130.2 to 64.1 G/s, and degree 40 drops from 54.5 to 33.0 G/s.
The spill footprint is 1.7 MB at 8,000 nodes and 4.4 MB at 11,582 nodes with 48 models, on either side of the 4 MB L2.
In plain order the same step is mild (118.0 to 96.6 G/s at degree 20) and no separate cliff appears.
Neighbours in plain order sit close in index, so spilled spin words reuse cache lines.

At one model, plain order shows no L2 exit up to 200,000 nodes. Throughput is flat from 100,000 to 200,000.
Shuffled order at one model drops at 200,000 nodes: 2.34 to 1.87 G/s at degree 20 and 1.34 to 1.00 G/s at degree 40.
The spill buffer is 3.1 MB at that size, and topology arrays compete for the same L2.
This is the only L2 exit seen at one model, and it sits just under the predicted 250,000.

**2. Cost of spill alone.** Moving the last 1,791 spins of a 5,791-node model out of shared memory costs 1.29x at one model and 2.03x at 48 models.
Further reduction of K adds at most another 0.5x, within the cell-to-cell spread of this grid.

**3. Spin traffic and topology traffic.** Compare 48 models at 200,000 nodes.
Doubling the degree from 20 to 40 raises the per-sweep time from 22.1 ms to 54.4 ms, which is 2.46x for 2x the edges.
At N = 5,791 with no spill, the same step costs 1.85x (180.7 to 334.1 µs).
Shuffling the labels changes only where the neighbours of a spilled spin sit.
It raises the time from 22.1 ms to 58.3 ms (2.64x) at degree 20 and from 54.4 ms to 115.0 ms (2.11x) at degree 40.
Spin placement and locality matter as much as doubling the topology.
At one model the shuffled penalty is smaller: 1.35x at degree 20 and 1.18x at degree 40 at 200,000 nodes.

**4. Cache evidence.** See the cache profile table.
L1 hit rate stays near 50% once N exceeds 5,791. The L2 hit rate falls with N and with model count.
DRAM throughput peaks at 25.5% of peak.

**5. Host side at the ceiling.** The kernel takes 80% of the 17.4 s model total, scoring takes 15%, and upload takes 3.5%.

**Hypothesis.** The prediction held for the L2 exhaustion at 48 models (between 8,000 and 11,582 nodes with shuffled labels).
The data does not support the claim that DRAM traffic at the ceiling is topology-dominated.
Shuffling, which alters only spin access, costs as much as doubling the degree.
DRAM reaches only 25.5% of peak at the ceiling, so the A4000 is not bandwidth-bound there.
`ncu` here does not split DRAM bytes by array, so the byte shares of spin and topology traffic are not measured.
The comparison in item 3 rests on timings.

Throughput at 200,000 nodes and 4,000,000 edges is 22.59 G/s for 48 models and 1.18 G/s for one model in plain order.
That is 21.2% and 44.0% of the all-shared degree-40 peaks at N = 5,791 (106.50 and 2.68 G/s).
In shuffled order the 48-model figure is 10.68 G/s, or 9.7% of the shuffled all-shared peak (110.18 G/s).

## Memory footprint

The session allocates, per `SelfFeedingSession::build`:
`col_ind` (4 bytes per directed edge), coefficients (3 slots per model, one value per directed edge for the unit lane in 1 byte and for the weighted `i32` lane in 4 bytes), fields, packed samples (3 slots × 128 reads × N/8 bytes per model), and the spill words (8 bytes each).
At 200,000 nodes and degree 40 there are 8,000,000 directed edges and 144 slots at 48 models.

| Lane | Computed (MiB) | Measured peak (MiB) |
| --- | ---: | ---: |
| Unit | 1,739 | 1,957 |
| Weighted `i32` | 5,117 | 5,317 |

The computed values exclude the CUDA context.
A one-model cell at the same size computes to 66 MiB and measures 293 MiB, so the context and module cost about 227 MiB.
Adding that figure gives 1,966 MiB for the unit lane and 5,344 MiB for the weighted lane, within 0.5% of the measured peaks.
The measured peak is the highest `nvidia-smi` reading, taken every 500 ms, in the time window of each cell.
Weighted `i32` at 48 models uses 5.2 GB of the card's 16 GB, almost all in per-slot coefficients.

## Limits of this measurement

- The graphs are synthetic circulant graphs with unit couplings and no fields. Real topologies have different locality.
- One device ran the grid. Another A4000 or a different clock state may give other numbers.
- Repeats use a fixed seed, so they test timing spread only.
- Cell-to-cell spread at one model reaches about 30% (see the shared fraction table).
- `ncu` runs 16 sweeps, with serialized kernels and fixed clocks, so its hit rates apply to the profiled run only.
- The grid does not test node counts between 5,791 and 8,000 or between 11,582 and 16,000.
- No weighted run at degree 20 exists, so the weighted reload cost is not separated.

## Reproduction

The evidence is under `/zfspool/quip-mining/bench-evidence/msa-dram-spill/` and stays outside the repository.
The directory holds `grid.sh`, `ncu.sh`, `summarize.py`, `grid.log`, `summary.md`, `memory.csv`, `device.txt`, the raw `grid/` and `ncu/` outputs.

```sh
docker run --rm -e CI=1 -v "$PWD":/src -v quip-cuda-cargo:/root/.cargo/registry \
  -v quip-cuda-target:/src/target -w /src quip-solver-cuda-ci cargo build --release --bins
ev=/zfspool/quip-mining/bench-evidence/msa-dram-spill
nvidia-smi --query-gpu=memory.used --format=csv -i 1 -lms 500   # optional sampler
$ev/grid.sh "$PWD"
$ev/ncu.sh "$PWD"
python3 -I $ev/summarize.py $ev > $ev/summary.md
```

Run the commands from the worktree at commit `0cdfa73`.
The `ncu` runs need `--cap-add=SYS_ADMIN` on the container, because `RmProfilingAdminOnly` is 1 on the host.
