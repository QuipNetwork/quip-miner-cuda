# Exact-coefficient corpus check

The unit mining paths pass the timing and energy checks against `main`.
The run covers `quip-cuda-sa`, `quip-cuda-gibbs`, and `quip-cuda-msa`.
The MSA run uses the default `auto` path.

## Setup

Device 0 is an `NVIDIA RTX A4000`.
Its compute capability is 8.6. Its power limit is 140 W.
The driver is 610.57.04. The `quip-solver-cuda-ci` image supplies CUDA 12.9.1.
The run date is October 12, 2026, in `US/Eastern` time.

| Tree | Commit | Cargo target volume |
| --- | --- | --- |
| Base (`main`) | `8826ca3` | `quip-cuda-target-base` |
| Branch | `298a978` | `quip-cuda-target` |

The base uses a detached worktree at `/tmp/qui1641-task12-base`.
Both commits support `bench run --source` and `--topology`.
The base lacks the `--msa-path` flag. Its MSA command omits that flag.

The eight nonces redraw real ±1 mining models.
Each model uses the Advantage2 system 1 topology.
The topology has 4,577 nodes and 41,515 edges, with degree at most 20.
No mining corpus exists under `quip-data`.
The input paths and SHA-256 hashes are:

```text
/zfspool/quip-mining/bench-evidence/qui1641-msa-weighted-lanes/corpus.jsonl
c0e88ba79af73075765a4ae60b861218c37ebfbc9c3696ef44439d0a19fad7e0
/zfspool/quip-mining/isingmark/fixtures/advantage2-system1.spec.json
f72af98b2bd6c1217d6d3c6389b3ba3e2790274d9e97822945c9f65b701ff4d3
```

See the [weighted lane benchmark](2026-10-msa-weighted-lanes.md) for the corpus nonces.

## Method

Each model uses 64 reads and 2,048 sweeps.
Each beta step takes four sweeps.
Each model has one warm-up and five measured repeats.
All runs use device 0 with the default `params.seed = 0`.
Each GPU run ends before the next run starts.

Average each repeat's CUDA-event sweep times across all eight models.
Take the median of those five averages.
The spread is the largest average minus the smallest average.
These times exclude builds, data transfers, and host scoring.
No `bench fold` step runs.

Timing passes if the absolute median change fits the base spread.
For a spread below 2% of the base median, timing can instead pass within 2%.
Use the spread and percentage change columns to check either rule.

The base benchmark discards the returned energies.
It writes no `best_energy` metric.
The probe links the base library without edits.
It calls `bench_one` with the same reads, sweeps, sweeps per beta, and seed.
It uses the same warm-up and repeat counts.
The probe keeps the lowest consensus-scored integer energy for each repeat.
The branch JSONL supplies its `best_energy` metric.
Probe code stays outside both trees.

The timing check applies the same rule to each model.
All 24 model timing checks pass.
Full model times and spreads are in `summary.json`.
Compare each model's energy first.
Use corpus means and minima only if models differ.
The five repeats keep the seed fixed. Repeats do not use new seeds.

## Corpus timing

Timing values are µs per sweep.

| Binary | Base median | Base spread | Branch median | Branch spread | Absolute change | Change % | Within spread | Within 2% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `quip-cuda-sa` | 10008.671 | 14.079 | 10025.113 | 3.073 | 16.443 | 0.164 | no | yes |
| `quip-cuda-gibbs` | 1534.230 | 1.616 | 1537.033 | 3.133 | 2.802 | 0.183 | no | yes |
| `quip-cuda-msa` | 111.343 | 0.435 | 110.833 | 0.043 | 0.509 | 0.457 | no | yes |

## Energy by model

Values are the best read energies in whole energy units. Lower is better.
Each column pair gives base / branch. The model order matches the corpus.

| Model | Nonce prefix | SA base / branch | Gibbs base / branch | MSA base / branch |
| ---: | --- | ---: | ---: | ---: |
| 0 | `6bd61f87d168` | −14802 / −14802 | −14790 / −14790 | −14816 / −14816 |
| 1 | `88cdbd8a8a9e` | −14739 / −14739 | −14705 / −14705 | −14733 / −14733 |
| 2 | `f25c20005121` | −14778 / −14778 | −14766 / −14766 | −14786 / −14786 |
| 3 | `e9fae0f36603` | −14776 / −14776 | −14768 / −14768 | −14780 / −14780 |
| 4 | `567eb89d7f41` | −14731 / −14731 | −14695 / −14695 | −14723 / −14723 |
| 5 | `6b0abda9e51c` | −14786 / −14786 | −14764 / −14764 | −14766 / −14766 |
| 6 | `98116f03b7a2` | −14771 / −14771 | −14787 / −14787 | −14775 / −14775 |
| 7 | `02fa84af47b6` | −14736 / −14736 | −14712 / −14712 | −14738 / −14738 |

Every model's base and branch best energies match.
This holds for each binary and all five repeats.
The energy check passes without a distribution fallback.

## Synthetic architecture benchmark

Both trees run `make bench-arch`.
It uses SA on a 512-node unit ring.
The run has eight reads, 1,024 sweeps, and five repeats.
Git ignores its output directory. The archive contains copies of the raw JSONL.
The original branch median is 3.082 µs slower than base, a 1.600% increase.
Its repeat range, 195.658–195.660 µs, does not overlap the base range, 192.576–194.140 µs.
The result passes the 2% timing rule, but that pass does not explain the slowdown.

| Base median µs | Base spread µs | Branch median µs | Branch spread µs | Absolute change µs | Change % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 192.577 | 1.564 | 195.659 | 0.002 | 3.082 | 1.600 |

### Interleaved timing check

The follow-up compares base `8826ca3` with branch `8cc5347` in six consecutive runs.
It rebuilds base in `quip-cuda-target-base` and reuses the branch build in `quip-cuda-target`.
Each run uses the same benchmark arguments, one warm-up, and five measured repeats.
Each run writes to a separate directory and uses a fresh `QUIP_CUDA_CACHE` directory.
The sweep median and spread come from the five CUDA-event measurements.
The spread is the largest measurement minus the smallest.

The table shows device 0 clock and temperature snapshots before → after each run.
These snapshots do not measure clocks during every repeat.

| Run | Tree | Median µs | Spread µs | SM clock MHz | Temperature °C |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | Base | 192.580 | 5.477 | 210 → 1905 | 59 → 62 |
| 2 | Branch | 192.581 | 0.007 | 1905 → 1905 | 62 → 63 |
| 3 | Base | 192.580 | 5.410 | 1905 → 1905 | 63 → 64 |
| 4 | Branch | 192.579 | 2.629 | 1905 → 1905 | 64 → 65 |
| 5 | Base | 192.579 | 1.076 | 1905 → 1890 | 65 → 66 |
| 6 | Branch | 194.110 | 6.567 | 1890 → 1890 | 66 → 67 |

Branch minus base is +0.001, −0.001, and +1.531 µs in the three adjacent pairs.
The last difference is +0.795%, within that pair's repeat spread.
The clock change predicts +0.794% for a workload whose time varies inversely with clock frequency.
The temperature rises with run order, and the clock drops before the last branch run.
At 1,905 MHz, the first two pairs show no consistent branch slowdown.
These results support an environmental explanation for the original gap, rather than a consistent branch-dependent difference.
The original run lacks matching clock snapshots, so the exact cause of its +1.600% increase remains unverified.

All before/after compute-process queries contain no process entries.
The archive includes SM and memory clocks, temperature, power draw, `utilization.gpu`, and process snapshots for both GPUs.
Device 0 memory clock rises from 405 to 7,001 MHz in the first run and stays at 7,001 MHz.

The SA int8 PTX has the same SHA-256 hash in all six fresh caches:

```text
ecec42285664d67b079a53d4e6be81dd8d208a02c7714667c9dd6d36ce244e1d
```

Both trees launch grid `(1, 1, 1)` and block `(256, 1, 1)`, with `shared_mem_bytes = 0`.
Both pass the same 18 kernel arguments in the same order and types.
The ring uses eight reads, 512 nodes, 1,024 directed adjacency entries, and 64 packed bytes per read.
Both pass 256 beta steps, four sweeps per step, and kernel seed 1.
The branch selects int8 at scale 1,000 milli, so coefficients and the beta schedule match base.
No launch dimension or argument difference exists for this int8 SA benchmark.

Raw JSONL, logs, PTX, command exit codes, and `summary.json` are under the archive's `interleaved/` directory.
Its `run.py` records the six runs, and `audit.py` records each command and exit code.
The follow-up removes the base worktree after the comparison.

## Evidence and reproduction

The archive holds raw JSONL and run logs.
It also holds input hashes, device data, image metadata, the probe, and scripts:

```text
/zfspool/quip-mining/bench-evidence/qui1641-corpus-regression/
```

`run.sh` builds both trees in Docker and runs the benchmarks under a taskd job.
`base-energy.rs` supplies the missing base energy measurements.
`clean-energy.py` keeps container startup text in the full stdout logs.
It writes the probe's JSON records to separate JSONL files.
`reduce.py` derives `summary.json` from the JSONL.
The archived commands use separate base and branch cargo target volumes.
The task removes the base worktree after the run.
Recreate that tree before another run:

```sh
git -C /zfspool/quip-mining/quip-solver-cuda worktree add --detach /tmp/qui1641-task12-base 8826ca3
```

Inside the container, the branch SA command is:

```sh
/src/target/release/quip-cuda-sa --device 0 bench run --source /corpus/corpus.jsonl --topology /topology.json \
  --reads 64 --sweeps 2048 --sweeps-per-beta 4 --warmup 1 --repeats 5 \
  --out /evidence/branch-sa
```

Only branch MSA adds `--msa-path auto`.
The container mounts the archived corpus at `/corpus` and the real topology file at `/topology.json`.
The evidence directory mounts at `/evidence`.

## Limits

These results cover eight fixed unit models on one GPU model.
Each run times one model in isolation.
The run does not test equal sample bytes across GPU models.
It also does not measure the weighted paths.
The [weighted lane benchmark](2026-10-msa-weighted-lanes.md) covers the I32 and F32 lane comparison.

## Persistent stream throughput

The final stream runs pass the throughput gate for all three binaries.
The original branch slows MSA by 19.464%.
The host changes remove that slowdown.

The baseline is `8826ca3`. The original head is `5660186`.
The final runtime source is `0177673`. The stream test follow-up is `07b4f11`.
Base uses `quip-cuda-target-base`. Head uses `quip-cuda-target`.
Both builds use the same CI image and device 0.
The input files and hashes match the preceding corpus.

### Stream method

The published conformance driver has fixed fixtures.
Its public entry points accept no corpus or workload settings.
The core `driver` module provides a single solve.
A test driver acts as the coordinator.
It uses the same Unix-domain gRPC session.
It depends on published protocol and solver crates, all at version 0.0.3.

The test driver redraws the eight real models with the published ChaCha8 code.
It sends exact I32 milli coefficients and dense edge lists.
It sends ordinary sampling jobs through the real binary session.
The session drives the CUDA sampler's persistent `run_stream` path.
The driver keeps the credit window full and checks every refund.

Each job requests 64 reads and 64 configured sweeps.
Gibbs reports its normal doubled budget, 128 sweeps.
Each beta step takes four sweeps. The beta range stays unset.
The queue depth is three.
SA and MSA grant 96 initial credits. Gibbs grants 24.
These grants match between base and head.

Each SA or Gibbs run completes 240 jobs.
Each MSA run completes 2,400 jobs.
Three pairs reverse their order in the middle pair.
Three extra MSA pairs check energy variation across independent session seeds.
Those pairs keep the same workload and alternate run order.

Timing starts at the first job dispatch and ends at the final result.
It includes preparation, admission, uploads, GPU work, host scoring, packing, and gRPC delivery.
It excludes pre-Hello startup, shutdown, and the driver's separate energy check.
The driver verifies every returned read after timing ends.
It uses the published integer-milli consensus scorer.

The 64-sweep budget is the SDK default.
A longer pilot uses 2,048 sweeps.
It heats device 0 to 95 °C and makes each SA run take minutes.
A second pilot uses larger job counts with the default budget.
Both pilots remain in the archive.
The final counts keep all stream lanes active and limit test time.

Stopping the first pilot's task leaves its child driver alive.
That driver overlaps a later try.
Process snapshots expose the interference.
The report excludes those attempts.
The final driver checks for other compute processes before and after every run.
Each container has a unique Docker name.
Signal cleanup stops the container if the driver ends early.

### Throughput results

Values are medians of completed jobs per second.
The original comparison is `before`. The final comparison is `packed`.
SA and Gibbs use three pairs. Final MSA uses six pairs.
Take each head/base ratio before computing the paired spread.
The 95% paired interval uses Student's t with two or five degrees of freedom.
A run passes within paired noise or a slowdown of at most 2%.
Every final binary passes both rules.

| Stage | Binary | Base jobs/s | Head jobs/s | Median change | Paired changes |
| --- | --- | --- | --- | --- | --- |
| before | `quip-cuda-sa` | 27.749 | 27.882 | +0.480% | −0.203%, +0.480%, −0.840% |
| before | `quip-cuda-gibbs` | 54.839 | 54.544 | −0.537% | −0.635%, −0.537%, −0.047% |
| before | `quip-cuda-msa` | 216.879 | 174.665 | −19.464% | −20.009%, −19.464%, −19.649% |
| packed | `quip-cuda-sa` | 28.212 | 28.104 | −0.381% | −0.564%, −0.381%, +0.030% |
| packed | `quip-cuda-gibbs` | 57.667 | 57.799 | +0.230% | +0.334%, +0.158%, −0.250% |
| packed | `quip-cuda-msa` | 218.920 | 219.192 | +0.124% | +0.280%, −0.168%, −0.109%, +0.187%, +1.092%, +0.667% |

### Host cost and fix

A normal job has one admission plan.
Upload originally converts its coefficients again.
The profile times actual planner and packing calls across the eight models.
Each stage has seven timed samples after warm-up.

| Stage | Original µs/job | Final µs/job |
| --- | ---: | ---: |
| SA plan | 421.516 | 428.621 |
| Gibbs plan | 420.033 | 27.671 |
| MSA plan | 450.816 | 28.152 |
| Coefficient packing | 558.798 | 102.318 |

Default beta resolution originally costs another 145.238 µs per job.
The final session caches integer topology degrees and the number of connected nodes.
It finds the unit model's largest field and gap count from each job's fields.
Tests compare the result bit for bit with the SDK across 3,888 field and sign cases.

The planner checks literal unit coefficients once.
It passes the result to session matching.
Gibbs and MSA can select their unit arithmetic without building milli vectors.
SA keeps its full local-field checks.
Packing selects one converter for the complete model, outside the CSR loop.
The unit converter keeps the target type check.
Other models keep the full milli, scale, and range checks.

Intermediate MSA changes leave slowdowns of 9.157%, 4.316%, and 3.842%.
The archive retains those results.
The final packing change brings MSA within the gate.
The source changes leave kernel code and launch parameters unchanged.

### Energy quality

The SDK calls `os_seed` for session jobs.
The persistent kernel carries its random state across jobs.
Treat independent runs as replicates.
Pooling jobs as independent trials understates the variation.

Each row below compares the mean of per-job best energies for one model.
Values are whole energy units. Higher means worse.
The interval half width comes from paired run means.
SA and Gibbs have three run pairs. MSA has six.
All returned read energies match independent consensus scoring.
Corpus mean differences fit paired run noise.
Every higher model mean fits its paired interval.
Gibbs model 2 improves beyond its unadjusted interval.
The raw records keep every job's best energy.

| Binary | Model | Base mean | Head mean | Head minus base | 95% half width | Within paired noise |
| --- | --- | --- | --- | --- | --- | --- |
| sa | 0 | −14491.911 | −14489.222 | +2.689 | 21.133 | yes |
| sa | 1 | −14428.267 | −14428.644 | −0.378 | 4.835 | yes |
| sa | 2 | −14461.333 | −14453.244 | +8.089 | 29.155 | yes |
| sa | 3 | −14458.356 | −14457.267 | +1.089 | 13.336 | yes |
| sa | 4 | −14416.267 | −14413.822 | +2.444 | 8.518 | yes |
| sa | 5 | −14441.089 | −14439.267 | +1.822 | 32.656 | yes |
| sa | 6 | −14440.667 | −14435.733 | +4.933 | 13.804 | yes |
| sa | 7 | −14385.444 | −14385.244 | +0.200 | 16.967 | yes |
| gibbs | 0 | −14518.089 | −14511.333 | +6.756 | 49.241 | yes |
| gibbs | 1 | −14455.289 | −14444.800 | +10.489 | 21.728 | yes |
| gibbs | 2 | −14474.067 | −14482.978 | −8.911 | 7.122 | no |
| gibbs | 3 | −14486.756 | −14474.822 | +11.933 | 51.489 | yes |
| gibbs | 4 | −14441.422 | −14445.422 | −4.000 | 22.343 | yes |
| gibbs | 5 | −14468.800 | −14465.489 | +3.311 | 23.228 | yes |
| gibbs | 6 | −14471.711 | −14471.289 | +0.422 | 10.648 | yes |
| gibbs | 7 | −14406.867 | −14407.711 | −0.844 | 25.992 | yes |
| msa | 0 | −14488.509 | −14490.556 | −2.047 | 13.920 | yes |
| msa | 1 | −14428.649 | −14423.033 | +5.616 | 13.195 | yes |
| msa | 2 | −14461.586 | −14459.830 | +1.756 | 16.098 | yes |
| msa | 3 | −14458.702 | −14458.619 | +0.083 | 3.276 | yes |
| msa | 4 | −14422.312 | −14415.973 | +6.339 | 10.957 | yes |
| msa | 5 | −14436.016 | −14444.181 | −8.166 | 9.170 | yes |
| msa | 6 | −14435.478 | −14438.194 | −2.717 | 12.912 | yes |
| msa | 7 | −14388.167 | −14385.210 | +2.957 | 16.222 | yes |

The preceding single-shot results also keep identical energies for every model with fixed seeds.
Session results use independent OS seeds and do not require identical sample bytes.

### Clock and temperature records

These are device 0 snapshots before and after each complete run.
They do not measure the average clock during the timed interval.
The archive keeps memory clocks, power, GPU load, and both cards' process lists.
Every final run starts and ends with no other compute process on device 0.

| Stage | Binary | Pair | Tree | Jobs/s | SM MHz before → after | °C before → after |
| --- | --- | --- | --- | --- | --- | --- |
| before | sa | 1 | base | 28.109 | 210 → 1560 | 68 → 78 |
| before | sa | 1 | head-before | 28.052 | 1560 → 1560 | 77 → 83 |
| before | sa | 2 | head-before | 27.882 | 1560 → 1860 | 83 → 87 |
| before | sa | 2 | base | 27.749 | 1860 → 1845 | 87 → 88 |
| before | sa | 3 | base | 27.551 | 1560 → 1560 | 88 → 89 |
| before | sa | 3 | head-before | 27.320 | 1560 → 1845 | 89 → 90 |
| before | gibbs | 1 | base | 54.740 | 1560 → 1560 | 89 → 89 |
| before | gibbs | 1 | head-before | 54.393 | 1560 → 1560 | 88 → 89 |
| before | gibbs | 2 | head-before | 54.544 | 1560 → 1560 | 88 → 89 |
| before | gibbs | 2 | base | 54.839 | 1560 → 1560 | 88 → 88 |
| before | gibbs | 3 | base | 54.917 | 1560 → 1560 | 88 → 88 |
| before | gibbs | 3 | head-before | 54.891 | 1560 → 1560 | 87 → 88 |
| before | msa | 1 | base | 219.031 | 1560 → 210 | 87 → 74 |
| before | msa | 1 | head-before | 175.205 | 210 → 210 | 74 → 71 |
| before | msa | 2 | head-before | 174.665 | 210 → 210 | 71 → 70 |
| before | msa | 2 | base | 216.879 | 210 → 210 | 70 → 69 |
| before | msa | 3 | base | 216.861 | 210 → 210 | 69 → 68 |
| before | msa | 3 | head-before | 174.251 | 210 → 210 | 68 → 68 |
| packed | sa | 1 | base | 28.340 | 210 → 1905 | 53 → 63 |
| packed | sa | 1 | head-packed | 28.180 | 1560 → 1875 | 63 → 70 |
| packed | sa | 2 | head-packed | 28.104 | 1560 → 1875 | 69 → 74 |
| packed | sa | 2 | base | 28.212 | 1560 → 1875 | 74 → 78 |
| packed | sa | 3 | base | 28.028 | 1560 → 1860 | 78 → 80 |
| packed | sa | 3 | head-packed | 28.037 | 1560 → 1560 | 79 → 82 |
| packed | gibbs | 1 | base | 57.629 | 1560 → 1560 | 81 → 81 |
| packed | gibbs | 1 | head-packed | 57.822 | 675 → 1560 | 81 → 82 |
| packed | gibbs | 2 | head-packed | 57.799 | 1560 → 1560 | 81 → 82 |
| packed | gibbs | 2 | base | 57.708 | 1560 → 1560 | 82 → 82 |
| packed | gibbs | 3 | base | 57.667 | 1560 → 1560 | 82 → 83 |
| packed | gibbs | 3 | head-packed | 57.523 | 630 → 1560 | 82 → 83 |
| packed | msa | 1 | base | 219.472 | 1560 → 210 | 82 → 70 |
| packed | msa | 1 | head-packed | 220.086 | 210 → 210 | 70 → 67 |
| packed | msa | 2 | head-packed | 219.210 | 210 → 210 | 67 → 66 |
| packed | msa | 2 | base | 219.579 | 210 → 210 | 66 → 65 |
| packed | msa | 3 | base | 218.369 | 210 → 210 | 65 → 65 |
| packed | msa | 3 | head-packed | 218.131 | 210 → 210 | 65 → 65 |
| packed | msa | 4 | base | 219.506 | 210 → 210 | 66 → 67 |
| packed | msa | 4 | head-packed | 219.915 | 210 → 210 | 63 → 66 |
| packed | msa | 5 | base | 216.808 | 210 → 210 | 67 → 68 |
| packed | msa | 5 | head-packed | 219.174 | 210 → 210 | 68 → 68 |
| packed | msa | 6 | base | 215.521 | 210 → 210 | 68 → 68 |
| packed | msa | 6 | head-packed | 216.960 | 210 → 210 | 68 → 68 |

### Stream evidence

The complete evidence directory is:

```text
/zfspool/quip-mining/bench-evidence/qui1641-stream-throughput/
```

`harness/` contains the Cargo project and lockfile.
`run.py` records commands, exit codes, raw output, and device snapshots.
`reduce.py` computes rates and paired noise.
`tables.py` computes energy comparisons from independent run means.
`host-profile/` times actual host calls.
`before/` and `packed/` hold the accepted original and final runs.
Other directories hold pilots and intermediate changes.
The report records every excluded try and setup failure.

To repeat the final runs after recreating the base worktree and binaries:

```sh
uv run --no-project --python 3.13 /zfspool/quip-mining/bench-evidence/qui1641-stream-throughput/run.py packed
uv run --no-project --python 3.13 /zfspool/quip-mining/bench-evidence/qui1641-stream-throughput/run.py packed --algorithms msa --first-pair 4 --pairs 6
```

The base worktree uses `/tmp/qui1641-final-base`.
The final run removes it after validation.
These results cover one GPU model and eight fixed mining models.
They measure ordinary sampling sessions without lease generation or target feedback.
They do not measure weighted-model throughput.
