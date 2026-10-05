# MSA on Turing: RTX 2070 results

Results from 2026-09-18 to 2026-10-05, measured on one RTX 2070 mining Aglais. They cover three
builds:

- the one-word build on this branch;
- the team's one-word fallback (MR !30), shipped in quip-miner-cuda v0.3.4-rc2 and v0.3.4-rc3;
- that fallback running under the quip-miner v0.3.5-rc9 and v0.3.5-rc11 coordinators.

## Setup

- **GPU:** RTX 2070 (Turing, compute capability 7.5, 8 GB). The opt-in shared memory per block
  is 64 KB.
- **Driver:** nvidia-driver-595-open 595.91.07.
- **Host:** Ryzen 7 3700X with 16 threads.
- **Stack:** the nodes.quip.network compose stack with the `cuda` profile, its own validator,
  on Aglais runtime 118.

| Build | quip-cuda-msa | Coordinator | Work |
|---|---|---|---|
| One-word build (this branch) | 0.3.4-rc1+msa1w, protocol 1 | 0.3.4 | plain jobs |
| Image v0.3.5-rc9 | v0.3.4-rc2, sha256 `38f75a53…e992`, protocol 1 | 0.3.5-rc9 | plain jobs |
| Image v0.3.5-rc11 | v0.3.4-rc3, sha256 `904cc3ce…b7fa`, protocol 2 | 0.3.5-rc11 | salt leases |

## 1. The one-word fallback works on Turing

v0.3.4-rc1 refused to open on this card: "requested 5000 nodes exceeds the device
shared-memory budget of 3551", exit 69. At one replica word the same card holds 7,102 spins.

**Startup, with nothing pinned (v0.3.4-rc3).** The probe chooses one word:
`msa session … num_colors=8 words=1 max_deg=20 shared_bytes=45320`.

- At target −14,619.92, adapt picks 64 reads × 29,529 sweeps.
- At live targets it picks 64 × 29,568.
- There were no TooLarge rejects and no NVRM Xid.

**Quality.** Each build replayed 144 live Aglais nonces, from qblocks 3470–3481. The score
is the best energy of 64 reads. Reads were pinned at 64 and sweeps at 29,568, and each build
ran twice.

| Comparison | Mean difference | 90% CI | Result (margin ±10) |
|---|---|---|---|
| v0.3.4-rc2 − one-word build (2026-09-19) | −0.61 | [−1.41, +0.17] | equivalent |
| v0.3.4-rc3 − one-word build (2026-10-05) | −0.24 | [−1.01, +0.51] | equivalent |
| v0.3.4-rc3 − v0.3.4-rc2 | +0.37 | [−0.47, +1.21] | equivalent |

Per-nonce energies correlate at 0.992 across builds. That confirms both builds solved the same
problems.

**Live mining.**

| | One-word build | Image v0.3.5-rc9 (rc2 msa) |
|---|---|---|
| Window | 2026-09-26 to 10-03, 1,336 rounds | 2026-10-05 04:11–06:11Z, 13 full rounds |
| Attempts per second | 8.11 (p10–p90 8.05–8.17) | **8.11** (8.10–8.13) |
| GPU power | ~160 W | ~168 W |
| Restarts, Xid, errors | none | none |

## 2. Lease mode mines 43% slower on this card

**Live, image v0.3.5-rc11 (rc3 msa, salt leases), 2026-10-05 06:14–08:14Z, 16 full rounds.**

| | v0.3.5-rc9, plain jobs | v0.3.5-rc11, leases |
|---|---|---|
| Attempts or salts per second | 8.11 | **4.64** (4.60–4.77) |
| GPU device time per job or salt | ~4.4 s | 3.94 s |
| GPU power | ~168 W | ~119 W |

Each salt runs slightly faster on the GPU, yet throughput falls and so does power. So the GPU
is waiting between salts rather than working harder.

**Offline, same binary.** Same rc3 binary and rc11 `quip-coordinator drive`, run back to back
at target −14,728:

| Run | Rate |
|---|---|
| Plain jobs, 432 | 7.90 /s |
| One 432-salt lease | 4.64 /s |
| Plain jobs, 432 | 7.87 /s |

Five leases of 64 salts gave 4.74 /s, so neither lease size nor GPU warm-up explains the drop.

**Likely cause, from reading the source; not proven.**

- In quip-solver-core 0.0.2, `crates/quip-solver-core/src/session.rs:1052`
  (`let slots = Arc::new(Semaphore::new(width));`) caps lease salts in flight at the stream
  width (36 on this card), with no prefetch.
- Plain jobs get a prefetch of twice the width. The comment at `session.rs:1009-1016` explains
  why: without the look-ahead, a lane stalls every cycle waiting for its next job.
- If that is the cause, it affects every GPU, not only Turing.

**Log behaviour in lease mode.** Over 2 h live:

- `lease issued` logged 1,386 times and `lease done` 0 times at INFO. Leases are cancelled at
  each round turnover.
- The `salts_per_sec` field in `lease issued` swings up and down instead of tracking the
  actual rate.
- The reliable throughput signal is attempt rows: one row per verified salt.

## 3. Latent: a failed probe falls back to 128 reads

In v0.3.4-rc3, `src/bin/quip_cuda_msa.rs:106` reads:

```rust
let words = probe_msa_replica_words(cli.device, nodes).unwrap_or(MSA_REPLICA_WORDS);
```

If the probe fails on a host that has a driver, the binary declares two words (128 reads). (The
host-without-a-driver case has been guarded since rc3.) On a 64 KB card every job is then
rejected TooLarge, and nothing reports an error. The coordinator's throughput heartbeat counts
the rejects as completed jobs. We did not hit this; it comes from reading the code.

## Context: winning

These are throughput and quality results. Aglais difficulty tightened from about −14,621 to
about −14,705 between 2026-09-21 and 09-26. Since then a 2070's best result per round falls
about 140–150 energy units short of the target. The one-word build won 8 qblocks between
2026-09-19 and 09-21 and none afterwards. None of the builds won during these tests.

## Method

- **Live throughput:** attempt rows from miner `cuda-0` in
  `data/attempts/<qblock>/attempts.jsonl`, counting rows with `is_pow` true and an empty
  `order_id`. Each round's count is divided by the time between its first and last row. The
  first and last round of each window are dropped. The shortfall is the per-round minimum of
  `raw_best_energy_milli − threshold_milli`.
- **Offline:** `quip-coordinator drive` on the same machine with the mining stack stopped and
  the GPU otherwise idle. Only one GPU process runs at a time.
- **Equivalence:** a two one-sided test with a ±10 energy-unit margin, and bootstrap 90%
  intervals on the paired per-nonce differences.
