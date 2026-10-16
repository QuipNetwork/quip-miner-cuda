# Lease screening results, October 2026

Use four probe reads, 512 probe sweeps, and a keep rate of 1/16.
Audit one in 200 rejects.
These defaults apply to multi-spin annealing (MSA) leases.
The timing pair recorded a target-hit rate ratio of 6.7115.
Direct shared device 0, so this pair does not establish an isolated throughput gain.

## Setup

| Setting | Value |
| --- | --- |
| Card | NVIDIA RTX A4000, 48 SMs, 16,376 MiB |
| PCI bus | `00000000:03:00.0` |
| Compute capability | 8.6 |
| Driver | 610.57.04 |
| Toolkit image | `quip-solver-cuda-ci`, CUDA 12.9.1 |
| Solver core | 0.0.3 |
| Corpus source commit | `555c63ee9f944b5a954f72d31a15baa15a18623f` |
| Cherry-picked corpus commit | `ccf4642` |
| Defaults commit | `38be02f` |
| Session test commit | `543ff5c` |
| Topology | Aglais, 4,577 nodes, 41,514 edges |
| Fields and couplings | Zero fields, couplings −1,000 and +1,000 milli |
| Full solve | 128 reads × 29,568 sweeps |
| Capacity | Production MSA capacity of 5,000 nodes, two replica words |
| Schedule | Geometric, automatic beta endpoints, one sweep per beta rung |
| GPU selection | Device 0, `CUDA_VISIBLE_DEVICES=0` |

Device 1 runs the production miner and takes no part in these runs.
Host checks hide both devices with `CUDA_VISIBLE_DEVICES=`.
The runner mounts only device 0 into its container.
It starts no MPS service.

Evidence lives in `/zfspool/quip-mining/bench-evidence/lease-screen/`.
The source fixture is `tests/fixtures/aglais.spec.json`.
Both programs preserve its edge order and use the protocol generator.

## Corpus and selection rule

The corpus contains 4,096 salts with a fixed nonce domain.
The last proof hash contains 32 bytes of `0x11`, and the miner account contains 32 bytes of `0x22`.
The base salt contains 32 bytes of `0x33`.
Each index replaces bytes zero through seven with its little-endian representation.

Full passes use seed inputs `0x123456789abcdef0` and `0xfedcba9876543210`.
The production streaming path also mixes wall time into its session seed.
The second full pass measures repeat variation on the same nonce corpus.

The corpus trial uses the hardest target, `i64::MIN`, to resolve the full-solve budget of 29,568 sweeps.
The session runs use the corpus percentile target and explicit coordinator read and sweep overrides.
Each pushed result must report 128 reads and 29,568 sweeps.

Let D be the keep denominator. For each probe shape and D, keep exactly floor N/D salts.
Sort by probe energy, then index.
Define the deepest sets from full_a alone, with the same index tie break.
Spearman correlation uses mean ranks for tied energies.

Choose the highest modeled deep-candidate rate among settings with deepest-50 recall of at least 90%.
For probe cost p, full cost f, and deepest-50 recall r, the formulas are:

- Cost per kept salt: p × D + f.
- Modeled deep candidates/s: r × 50/N ÷ (p + f/D).
- Unscreened deep candidates/s: 50/N ÷ f.

Probe cost excludes setup and warm-up.
Full cost includes host drawing, transfers, exact scoring, session setup, and teardown.
The modeled rate measures corpus recall and cost, rather than coordinator target hits.

## Corpus timings

The following tables copy the task 5a report.

| Phase | Wall seconds |
| --- | ---: |
| Device and corpus setup | 2.338627 |
| Probe 4 × 256, setup | 0.768882 |
| Probe 4 × 256, warm-up | 0.032701 |
| Probe 4 × 256, corpus | 2.332061 |
| Probe 4 × 512, setup | 0.014089 |
| Probe 4 × 512, warm-up | 0.070971 |
| Probe 4 × 512, corpus | 9.090639 |
| Probe 4 × 1024, setup | 0.014302 |
| Probe 4 × 1024, warm-up | 0.141335 |
| Probe 4 × 1024, corpus | 18.117167 |
| Probe 8 × 512, setup | 0.017109 |
| Probe 8 × 512, warm-up | 0.142895 |
| Probe 8 × 512, corpus | 18.260225 |
| Full_a | 392.079113 |
| Full_b | 400.096172 |
| Total in example | 843.529112 |

## Corpus results

Corpus: 4096 nonces. Full-versus-full Spearman: 0.984380.
Full time: 96.701085 ms/nonce, mean of both full passes.
Unscreened deep candidates/s: 0.126235.

| Probe reads × sweeps | Spearman vs full_a | Probe ms/nonce |
| --- | ---: | ---: |
| 4 × 256 | 0.813161 | 0.569351 |
| 4 × 512 | 0.842395 | 2.219394 |
| 4 × 1024 | 0.879305 | 4.423137 |
| 8 × 512 | 0.875015 | 4.458063 |

| Shape | D | Kept | Deepest-10 recall | Deepest-50 recall | Cost ms/kept | Deep candidates/s | Rate / unscreened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| probe_2_256 | 16 | 256 | 90.0% | 74.0% | 105.810697 | 1.365942 | 10.8207 |
| probe_2_256 | 32 | 128 | 80.0% | 58.0% | 114.920310 | 1.971475 | 15.6175 |
| probe_2_256 | 64 | 64 | 80.0% | 46.0% | 133.139536 | 2.699236 | 21.3827 |
| probe_2_256 | 128 | 32 | 80.0% | 32.0% | 169.577987 | 2.948496 | 23.3573 |
| probe_2_256 | 256 | 16 | 60.0% | 18.0% | 242.454890 | 2.320019 | 18.3786 |
| probe_2_512 | 16 | 256 | 100.0% | 90.0% | 132.211395 | 1.329547 | 10.5323 |
| probe_2_512 | 32 | 128 | 100.0% | 74.0% | 167.721706 | 1.723465 | 13.6529 |
| probe_2_512 | 64 | 64 | 90.0% | 56.0% | 238.742326 | 1.832520 | 14.5168 |
| probe_2_512 | 128 | 32 | 50.0% | 36.0% | 380.783568 | 1.477217 | 11.7021 |
| probe_2_512 | 256 | 16 | 40.0% | 20.0% | 664.866052 | 0.940039 | 7.4468 |
| probe_2_1024 | 16 | 256 | 100.0% | 90.0% | 167.471270 | 1.049620 | 8.3148 |
| probe_2_1024 | 32 | 128 | 100.0% | 76.0% | 238.241455 | 1.246110 | 9.8714 |
| probe_2_1024 | 64 | 64 | 100.0% | 64.0% | 379.781825 | 1.316545 | 10.4293 |
| probe_2_1024 | 128 | 32 | 90.0% | 46.0% | 662.862565 | 1.084312 | 8.5897 |
| probe_2_1024 | 256 | 16 | 70.0% | 24.0% | 1229.024046 | 0.610240 | 4.8342 |
| probe_3_512 | 16 | 256 | 100.0% | 98.0% | 168.030090 | 1.139119 | 9.0238 |
| probe_3_512 | 32 | 128 | 100.0% | 76.0% | 239.359096 | 1.240291 | 9.8253 |
| probe_3_512 | 64 | 64 | 70.0% | 52.0% | 382.017108 | 1.063434 | 8.4243 |
| probe_3_512 | 128 | 32 | 60.0% | 36.0% | 667.333131 | 0.842907 | 6.6773 |
| probe_3_512 | 256 | 16 | 50.0% | 26.0% | 1237.965178 | 0.656319 | 5.1992 |

Rule: choose the highest deep rate with deepest-50 recall ≥ 90%.
Recommendation: `probe_2_512`, D=16. Deepest-50 recall is 90.0%. Expected deep rate is 1.329547/s.

Tied values share a mean rank. Keep counts use energy, then index, from low to high.
The deepest sets break ties by index, with no use of probe scores.
The keep count is floor N/D. Rate and cost use D as given.

The selected probe cutoff is −14,312,000 milli. Fourteen rows share that score, and the index tie break keeps nine.
None of those tied rows belongs to the deepest-50 set, so other tie orders leave its 90% recall unchanged.
The full_a deepest-50 cutoff is −14,514,000 milli, and all seven rows at that score fit the deepest-50 set.

The chosen shape meets the 90% recall rule with no margin.
The 8-read × 512 shape at D=16 reaches 98% recall but has a lower modeled rate, 1.139119/s.
These corpus rates come from one trial. The session check below tests the live rate.

## Chosen defaults

```toml
[screen]
enabled = true
probe_reads_log2 = 2
probe_sweeps = 512
keep_denominator = 16
audit_denominator = 200
```

This probe shape and keep rate give the highest rate under the stated rule.
The live selector learns its cutoff as it runs. Its keep rate can differ from 1/16.
Probe target hits always proceed to a full solve.
The selector also fully solves about one in 200 otherwise rejected salts for observation.
Audits compare full energy with the median of the most recent 256 kept full results.
Audit observations do not adjust the cutoff.

## Whole-pipeline method

`tests/common/lease_pipeline.rs` runs through the coordinator in `tests/common/lease.rs`.
The coordinator starts the production `quip-cuda-msa` binary and opens a solver-core session over a Unix socket.
Each mode receives the same topology, target, and nonce domain.
The screened mode sends an empty configuration, which exercises the defaults.
The Direct mode sends `[screen] enabled = false`.

The target is **−14,518,000 milli**.
The nearest-rank rule takes full_a[ceil(0.01 × 4096) − 1] after sorting, with zero-based indexing.
This is rank 41. Ties place 42 corpus salts at or below the target.
The target needs one solution and sets no diversity rule.
Coordinator overrides pin the full solve to 128 reads and 29,568 sweeps in both modes.

Each run starts its timer immediately before sending a lease for ten million salts.
The run collects traffic for ten nominal minutes, then cancels that lease.
The run includes first-job JIT and device setup after job submission.
Handshake and initial device opening precede the timer.
The coordinator requests `Status.jobs_done` at each minute boundary and records the actual elapsed time.
Status round trips can extend a boundary slightly.
Each rate uses the elapsed time in the log.

Completed salts include screened rejects and full results.
Core can count a result before its frame reaches the server.
Status and push snapshots can differ at a boundary.
A push is a full result, even when all its energies miss the target.
A target hit is a pushed result whose best energy is at or below the target.
The test asserts the shape and salt uniqueness for every push.
It independently verifies the nonce, coefficients, and all read energies for one result in each window.
Verification and CSV output occur between window collections and contribute to the session workload.
The raw CSV records each pushed salt index and its best energy.

Cancellation traffic has separate counters and does not enter the timed push or hit totals.
The final lease log records screened, kept, and audited decisions.
Kept and audited decision totals can include work still pending at cancellation.
Divide each decision count by screened + kept + audited to get its fraction.
They describe admission decisions, rather than completed full solves.

## Whole-pipeline results

Both runs use ten nominal minutes. The table uses the logged elapsed time.
**Direct shared device 0.** Its postflight found a foreign GPU process, PID 2946356, using 194 MiB.
The process name was `round4-cuda-harness`. The overlap duration is unknown.
The preflight found no compute process at 17:02:35 UTC.
A 17:10:47 UTC snapshot showed only the test miner. The foreign process appeared in the 17:12:37 UTC postflight.
Screened snapshots showed no foreign compute process.
The ratio below describes this pair and does not establish an isolated throughput gain.

| Measure | Screened defaults | Direct, shared |
| --- | ---: | ---: |
| Elapsed seconds | 600.001415 | 600.001278 |
| Completed salts | 44929 | 5232 |
| Pushes | 3088 | 5232 |
| Target hits | 349 | 52 |
| Salts/s | 74.881490 | 8.719981 |
| Pushes/s | 5.146655 | 8.719981 |
| Target hits/s | 0.581665 | 0.086666 |
| Best pushed energy, milli | -14628000 | -14594000 |
| Pushes during cancellation | 0 | 0 |
| Final completed salts | 44929 | 5232 |

The screened-to-Direct target-hit rate ratio is **6.7115**.
The Direct run shared device 0. This ratio needs a repeat with matching GPU conditions.

### Selection fractions

These counts include selected work still pending at cancellation.

| Decision | Screened count | Screened fraction | Direct count | Direct fraction |
| --- | ---: | ---: | ---: | ---: |
| screened | 41841 | 93.0710% | 0 | 0.0000% |
| kept | 2912 | 6.4774% | 5280 | 100.0000% |
| audited | 203 | 0.4516% | 0 | 0.0000% |

### Audit observations

The run logged 201 completed audits.
Of those, 201 had a kept median for comparison.
In 5 cases, the audit energy was lower than that median.
The median audit-minus-kept-median gap was 87000 milli.
A positive gap means that the audit energy was higher than the kept median.
The best audit energy was −14492000 milli.
Audits produced 0 target hits, including the cancellation tail if present.
The comparison uses the kept median at the time of each audit.
It does not use one median for the entire run.

### Per-minute series

Rates use each window's elapsed time. Counts reset at each row.

| Mode | Minute | Salts | Pushes | Hits | Salts/s | Pushes/s | Hits/s | Best energy, milli |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| screened | 1 | 4949 | 303 | 40 | 82.4809 | 5.0498 | 0.6666 | -14590000 |
| screened | 2 | 4353 | 306 | 37 | 72.5501 | 5.1000 | 0.6167 | -14574000 |
| screened | 3 | 4380 | 319 | 34 | 73.0001 | 5.3167 | 0.5667 | -14568000 |
| screened | 4 | 4768 | 275 | 32 | 79.4674 | 4.5834 | 0.5333 | -14556000 |
| screened | 5 | 4377 | 321 | 27 | 72.9498 | 5.3500 | 0.4500 | -14628000 |
| screened | 6 | 4363 | 311 | 35 | 72.7169 | 5.1834 | 0.5833 | -14578000 |
| screened | 7 | 4716 | 292 | 29 | 78.5991 | 4.8666 | 0.4833 | -14570000 |
| screened | 8 | 4324 | 324 | 40 | 72.0665 | 5.4000 | 0.6667 | -14580000 |
| screened | 9 | 4345 | 320 | 39 | 72.4174 | 5.3334 | 0.6500 | -14568000 |
| screened | 10 | 4354 | 317 | 36 | 72.5664 | 5.2833 | 0.6000 | -14594000 |
| direct | 1 | 576 | 576 | 4 | 9.5997 | 9.5997 | 0.0667 | -14528000 |
| direct | 2 | 528 | 528 | 8 | 8.8000 | 8.8000 | 0.1333 | -14564000 |
| direct | 3 | 528 | 528 | 2 | 8.7992 | 8.7992 | 0.0333 | -14564000 |
| direct | 4 | 576 | 576 | 6 | 9.6009 | 9.6009 | 0.1000 | -14538000 |
| direct | 5 | 544 | 544 | 7 | 9.0646 | 9.0646 | 0.1166 | -14552000 |
| direct | 6 | 512 | 512 | 9 | 8.5353 | 8.5353 | 0.1500 | -14568000 |
| direct | 7 | 528 | 528 | 3 | 8.7997 | 8.7997 | 0.0500 | -14594000 |
| direct | 8 | 528 | 528 | 5 | 8.8003 | 8.8003 | 0.0833 | -14546000 |
| direct | 9 | 551 | 550 | 4 | 9.1833 | 9.1666 | 0.0667 | -14582000 |
| direct | 10 | 361 | 362 | 4 | 6.0167 | 6.0334 | 0.0667 | -14524000 |

The JSON and push CSV files contain the counts and elapsed times behind these tables.

## Reproduction

The external runner records process snapshots before and after every GPU command:

```sh
bash /zfspool/quip-mining/bench-evidence/lease-screen/task5b-run.sh
```

The runner first checks screen and lease behavior, then runs eight-second trials.
It requires an empty device-0 process list before each ten-minute run.
A timed run inside the device-0 container uses:

```sh
CUDA_VISIBLE_DEVICES=0 LEASE_PIPELINE_OUT=/evidence \
  LEASE_PIPELINE_MODE=screened LEASE_PIPELINE_SECONDS=600 \
  cargo test --release --test lease_gpu pipeline::whole_pipeline -- \
  --ignored --exact --nocapture
```

Set `LEASE_PIPELINE_MODE=direct` for the comparison.
Place `corpus-4096.csv` in `LEASE_PIPELINE_OUT` so the test can compute the target.
The runner supplies `PIPELINE_COMMIT` and records the source commit in each JSON result.

## Evidence limits

The evidence covers one card, one topology, one corpus, and the full-solve budget of 29,568 sweeps.
The corpus selection has no holdout trial.
The chosen shape meets the 90% deepest-50 recall rule with no margin.
A coordinator target with a smaller solve budget can change the cost tradeoff.

The live cutoff differs from the corpus's fixed rank quota.
Audit selection uses fresh OS entropy, and full-solve seeds include wall time.
The modes process different numbers of salts, so the timing pair is not a paired per-nonce quality experiment.
One ten-minute trial per mode does not measure run-to-run variance.
Screened runs first. Direct follows in a fresh process, so run order can affect heat and clock state.

Process snapshots show GPU use at each check. They cannot prove sole use for the full run.
Task 5a reported unverified isolation for its cost model.
This check measures pushed target hits.
It does not measure chain rewards or proof acceptance.
