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
This benchmark also passes the 2% timing rule.

| Base median µs | Base spread µs | Branch median µs | Branch spread µs | Absolute change µs | Change % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 192.577 | 1.564 | 195.659 | 0.002 | 3.082 | 1.600 |

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
