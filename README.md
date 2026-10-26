# `quip-solver-cuda`

CUDA Ising miners for the [quip.network](https://gitlab.com/quip.network) v0.3 mining protocol.
Each solver has its own binary. See the list below. **amd64 only.**

Each process binds one CUDA device (`--device N`) and drives it directly.
The coordinator takes N from the `[cuda.N]` config section and passes it as
`--device N`. The miner warns at startup when its `cuda-N` label and `--device`
disagree, and the label never overrides the flag.
Solver kernels and the lease probe (`kernels/screen.cu`) are JIT-compiled via NVRTC at
runtime through `cudarc`'s `dynamic-loading` feature. **Building this crate
does not require the CUDA toolkit.** Running the binaries needs a CUDA GPU
and driver.

Supported GPUs have compute capability 7.0 (Volta) through 12.1 (consumer
Blackwell). The floor comes from the kernels (`__nanosleep` is sm_70+).
The ceiling comes from NVRTC 12.9 (`cuda-12090` in `Cargo.toml`).
The miner compiles kernels for each device's detected capability, clamped into
that range. A capability that the toolkit lacks, such as 8.8, gets the next
lower architecture and loads through the driver's forward-compatible PTX JIT.
`SUPPORTED_ARCHS` in
`src/cuda_device.rs` is the contract and `tests/arch_coverage.rs` enforces it
(`make test-archs`). The host scores energies with the canonical
`quip_protocol::scoring::energy_milli` so results match consensus.
`quip-cuda-msa` keeps as many spins as fit in the opt-in shared memory per
block and spills the rest to a per-nonce buffer in device memory. Node and
coupler limits come from device memory divided among instances,
one per SM. With 15 GiB free, 48 SMs and 128 reads, the A4000 advertises
3,987,504 uncoupled nodes. It also fits 44,486,028 dense couplers at 9,433 complete nodes.
These independent Binary limits use the default 64-beta schedule. A model near both limits can
exceed memory. The miner checks each model before allocation, including its
precision and schedule length. Shared memory sets how many spins stay on chip.
The miner refuses a session whose buffers exceed device memory. At 200,000 nodes
and 4,000,000 edges with 48 models, the A4000 measured about 2 GiB for unit
models and 5.2 GiB for weighted i32 models (see the
[A4000 spill measurements](docs/perf/2026-10-msa-dram-spill-a4000.md)).

Run `quip-cuda-msa --probe-capacity` to show the capacity grid for the selected device.
Add `--device N` to select a GPU or `--json` for machine-readable output.
The grid uses the mining read count and 64 beta entries.
It shows separate Binary, Int8, Int32, and Float32 limits. Int32 also counts its integer acceptance tables.
Each cell shows variables, couplers, and the shared, spill, or dense storage tier.
Use `quip-cuda-msa bench run --complete N --precision int8 --out DIR` to benchmark a complete graph.
The precision options are `binary`, `int8`, `int32`, and `float`.

## Binaries

| binary | algorithm |
|--------|-----------|
| `quip-cuda-sa` | simulated annealing (Metropolis) |
| `quip-cuda-msa` | multi-spin coded simulated annealing, 64 reads per word |
| `quip-cuda-gibbs` | heat-bath Gibbs |

Each [release](https://gitlab.com/quip.network/quip-solver-cuda/-/releases)
provides prebuilt `amd64` binaries.

## Model limits

All three binaries list `exact-coefficients` in `--capabilities`.
They refuse any model that they cannot represent exactly, with a capacity error.

The model scale is the greatest common divisor (`GCD`) of its nonzero coefficients.
The scale uses milli units.

At each spin, add `|h|` and all `|J|` on its edges.
The local field bound is the largest of these sums.

- **SA:** solves every whole-milli model up to a local field of 536,870,911 milli.
  SA uses the int8 kernel when `max_i(|h_i| + sum_j |J_ij|) / GCD_step <= 63`.
  This bound keeps the flip delta within int8 because `2 * field <= 127`.
  Other models use the fixed-point wide kernel.
  A complete graph with unit couplings and zero fields has field 63 at 64 nodes. It uses int8.
  At 65 nodes, its field is 64 and it uses wide.
  Its accept decision uses integers only, so the same seed gives the same decision on every GPU.
  The SA identity advertises 536,870,910 milli because `quip-solver-core` 0.0.3 `--solve` does not round coefficients (QUI-1813).
  The wide kernel cannot accept a move whose Metropolis probability is below about 7.6e-6.
- **Gibbs:** accepts models whose absolute coefficients are at most 127 times the model's GCD step.
  Gibbs refuses models that need a wide path.
- **MSA:** uses sparse storage through degree 62 and dense storage for higher degrees.
  Models with `h` in {-1, 0, 1} and `J` in {-1, 1} use the unit-weight kernel.
  Other models use the weighted kernel at the model's GCD step.
  Its local field bound is 16,777,215 times that step, or 16,777,215 milli at scale 1.
  The default weighted lane is I32. F32 is also available.
  See the [weighted lane measurements](docs/benchmarks/2026-10-msa-weighted-lanes.md).

The unit mining paths pass the timing and energy checks against `main` at `8826ca3` on the RTX A4000.
See the [base and branch comparison](docs/benchmarks/2026-10-exact-coefficients-corpus.md) for per-binary results and evidence.

## Build

```sh
cargo build --release
```

`cudarc` uses `dynamic-loading`, so the build links against no CUDA libraries.
The miner loads the CUDA driver and compiles kernels at process start.

[quip.network/quip-solver-core](https://gitlab.com/quip.network/quip-solver-core)
publishes the solver contract (`quip-proto`, `quip-protocol`, `quip-solver-core`) to crates.io.

## Running

Requires a CUDA-capable GPU and driver at runtime.

**Connect to a coordinator** (production):

```sh
quip-cuda-sa --quip-coordinator unix:///run/quip/coord.sock --device 0
```

**Driver / fixed-input (run in isolation).** Use the coordinator's
`drive` harness pointed at the binary — `--source random` for golden-drawn
problems, `--source list <jsonl>` for a fixed replay:

```sh
quip-coordinator drive --miner ./quip-cuda-sa \
  --source random --topology-preset advantage2-system1 \
  --count 8 --num-reads 16 --num-sweeps 1030 --report out.jsonl
```

**Introspection:**

```sh
quip-cuda-sa --capabilities   # capabilities JSON
quip-cuda-sa --check          # probe the backend is runnable
```

## Multi-spin kernel (`quip-cuda-msa`)

`kernels/msa.cu` is a CUDA port of the multi-spin coded simulated annealing
in `quip-solver-cpu`'s `quip-cpu-msa` (Isakov, Zintchenko, Rønnow, Troyer,
*Optimised simulated annealing for Ising spin glasses*, Comput. Phys. Commun.
192, 2015). 64 replicas share one 64-bit word per spin. A job of 128 reads is
two words. The Metropolis test is an integer comparison against a per-rung
threshold table. Spins update one color class at a time, with the same greedy
coloring the Gibbs kernel uses. One block anneals one problem in shared
memory with a global spill buffer. The full MSA kernel computes no energies. The host rescores every sample with
`energy_milli`, so results are consensus-scored like the other kernels.

The unit kernel uses couplings and fields by sign. Consensus problems have
`J` in {-1, 1} and `h` in {-1, 0, 1}, which it represents exactly.
Other supported whole-milli models use the weighted kernel at their own scale.
The miner refuses any model that exceeds the coefficient or local field bounds.
The CUDA MSA solver uses dense storage for spins with more than 62 neighbors.

The first K spins live in shared memory. To get K, subtract 8720 bytes from the opt-in shared memory per block.
Divide by 8 times the number of replica words. At 128 reads, two words use 16 bytes per spin.
A 99 KB part holds 5791 spins.
Turing uses 64 reads and holds 7102 spins in shared memory.
Spins past K live in a per-nonce global buffer,
which reaches L2 and then DRAM as the graph grows. Placement does not change
the samples: a spilled run returns the same reads as an all-shared run of the
same seed. Without `--max-nodes`, the miner uses its device-derived node limit.
The flag sets a lower cap. A value that exceeds the device limit fails at open.
The error names the largest capacity that fits. MSA `--capabilities`
opens the selected device to read its memory limits.
See the [A4000 spill measurements](docs/perf/2026-10-msa-dram-spill-a4000.md)
for the throughput at each size.

Two host-side changes came with it and apply to `quip-cuda-sa` too:

* **Stop on cancel.** When the coordinator cancels a round, in-flight models
  stop at the next rung instead of running to completion. The new round
  starts within milliseconds rather than after a full batch.
* **Parallel scoring.** The host rescores downloaded samples on `QUIP_SCORE_THREADS`
  threads (default 4). With one thread the sampler capped a fast kernel at
  about 20 jobs per second and left the GPU idle between batches.

## Lease screening (`quip-cuda-msa`)

MSA accepts coordinator salt leases and screens each salt before a full solve.
By default, the probe draws couplings on the device with ChaCha8 and runs
4 reads for 512 sweeps. An adaptive energy cutoff keeps about 1 in 16 salts.
Probe target hits always proceed. The miner also audits about 1 in 200 rejects.
Kept and audited salts receive a full MSA solve with exact host scoring.

The probe requires zero fields, both −1000 and +1000 milli couplings, degree
at most 20, and a topology that fits the device's shared memory.
Other supported models use Direct mode, which fully solves every salt.
This includes weighted models and the production topology on Turing's 64 KB budget.
SA and Gibbs solve jobs sent by the coordinator.

Plain stream sessions do not take the lease turn. A sustained full-width plain
stream on the same device can delay a lease probe and its cancellation until
the plain session drains.

The coordinator supplies these keys in the miner configuration:

| `[screen]` key | Default | Accepted values |
| --- | --- | --- |
| `enabled` | `true` | `true` or `false`. Set `false` for Direct mode. |
| `keep_denominator` | `16` | 2–100,000. Target keep fraction is 1/value. |
| `audit_denominator` | `200` | 0 or 2–1,000,000. Audit fraction of rejects is 1/value. Zero disables audits. |
| `probe_reads_log2` | `2` | 2–5, for 4–32 reads. |
| `probe_sweeps` | `512` | 1–2048. |

Invalid numeric values log an error and keep the previous value.
Audits observe full-solve quality but do not adjust the cutoff.

On an RTX A4000 with Aglais, a ten-minute trial found 6.7 times as many
target hits per second as Direct mode. Direct shared the GPU late in that trial.
The first eight minutes gave a ratio of 6.2.
See the [calibration and session results](docs/perf/2026-10-cuda-lease-screen.md) for methods and limits.

The probe kernel comes from rdyplayerB's
[QuipNetwork/quip-solver-cuda#1](https://github.com/QuipNetwork/quip-solver-cuda/pull/1).

## Sharing the graphics processor

`--yielding` lets the miner share a GPU. The NVML governor measures the load
from other processes and compares it against `--utilization`. It does not use
device-wide `utilization` for this decision, because that figure counts the
miner's own kernels. A busy miner holds device-wide `utilization` near 100
percent, so a governor that read it would throttle against itself.

When another process passes the ceiling, the miner ends its current session and
waits. Ending the session is what frees the SMs. The kernel is persistent and
holds its SMs until teardown, so a pause inside a session yields nothing.

Per-process attribution needs both NVML support and a process ID that matches.
Inside a container, or on WSL2, the governor falls back to device-wide
`utilization` and logs which method it uses. The fallback applies only while
another process holds a context. A miner alone on a GPU never throttles.

## Session recovery

Each device handle precreates 64 pairs of compute and transfer streams.
At most 64 sessions per handle run concurrently.
Further sessions wait for a pair to return after teardown drains both streams.
Session setup creates no streams while persistent kernels run.

The driver ends a self-feeding session and builds a new one in two cases.

The pipeline goes empty. Nothing is in flight and the queue holds nothing, so no
completion can arrive. The kernel is persistent and keeps every SM it launched
with, so a wait inside the session holds the device for no work. After the
pipeline stays empty for 2 seconds, the driver ends the session and parks in a
blocking receive. The next seed starts a new session.

No slot completes. The host waits for a slot to complete. The kernel waits for a slot to become ready.
If these states diverge, both sides wait. Each side behaves as written. Neither can detect the stall. The driver
measures the longest gap between completions and sets the cutoff at 4 times that
gap, with a floor of 10 minutes. Past the cutoff it rejects the jobs it holds,
which refunds their credits at the coordinator, and starts a new session. It
logs one line:

```text
quip-solver-cuda: no slot completed in 612.4s; rebuilding the self-feeding session
```

## Driver time budget

The stream driver can account for its own wall clock, one window at a time.
Use it to find where driver time goes when throughput falls.

Accounting is off by default. The table lists the environment variables.
`QUIP_SCORE_THREADS` applies to every binary whether accounting is on or off.

| Variable | Effect |
| -- | -- |
| `QUIP_DRIVER_BUDGET` | Set to `1` to turn accounting on. Any other value leaves it off. |
| `QUIP_DRIVER_BUDGET_WINDOW` | Report period in seconds. Default 60. |
| `QUIP_DRIVER_BUDGET_OUT` | Path to append one JSON object per window. Optional. |
| `QUIP_SCORE_THREADS` | Host threads that score one job's reads. Default 4. Values below 1 fall back to the default. |

Each window logs one line:

```text
[QUI-870 budget] win=12 up=48.0min att/s=2.60 | poll=3.1% ul=1.2% dl=8.4% score=6.0% consumer=71.0% throttle=0.0% spin=10.2% unacct=0.1%
```

The buckets are `poll` (ctrl mailbox reads), `ul` (slot uploads), `dl` (sample
downloads), `score` (host-side energy scoring), `consumer` (blocking sends to
the result channel), `throttle` (time yielded to another GPU user), and `spin`
(the idle backoff). `unacct` is window time that no bucket claimed.

Read the output by asking which bucket grows while `att/s` falls. A growing
`unacct` share is a result too. It means the cost sits outside every region the
driver measures.

### Soak test

`tests/driver_budget_soak.rs` drives the streaming loop for a set duration. It
records `att/s` alongside the NVML readings for board power, core clock, and temperature:

```sh
QUIP_DRIVER_BUDGET=1 QUIP_DRIVER_BUDGET_WINDOW=300 \
QUIP_DRIVER_BUDGET_OUT=/tmp/soak_budget.jsonl \
QUIP_SOAK_MINUTES=240 QUIP_SOAK_SAMPLE_SECS=120 \
QUIP_SOAK_SAMPLE_OUT=/tmp/soak_samples.csv \
cargo test --release --test driver_budget_soak -- --ignored --nocapture
```

Read the clock column before the throughput column. A card that throttles on
temperature loses core clock, and that loss looks like a software decay.

## Tests

```sh
cargo test --release                       # host-only tests
cargo test --release -- --include-ignored  # adds the CUDA-device tests
```

The ten-minute `pipeline::whole_pipeline` benchmark skips unless `LEASE_PIPELINE_OUT`
names a directory that contains `corpus-4096.csv` from the
[lease screen calibration](docs/perf/2026-10-cuda-lease-screen.md).

`tests/arch_coverage.rs` needs the pinned CUDA 12.9 toolkit, so it skips unless
the environment sets `CI`. Run it with `make test-archs`, which supplies both
the toolkit and the variable. On a different toolkit the test measures that
toolkit's own architecture support instead of the kernels, so a skip is the
correct result. Each skipped test prints a `SKIP` line.

Conformance/golden and handshake tests drive the binary in isolation via
`quip-solver-conformance`'s driver and check energies against its bundled
golden vectors.
Tests that need a live CUDA device carry `#[ignore]`, so a machine without a
GPU reports them as ignored. Run them with `--include-ignored`
on a CUDA host.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
