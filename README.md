# `quip-solver-cuda`

CUDA Ising miners for the [quip.network](https://gitlab.com/quip.network) v0.3
mining protocol: simulated annealing (`quip-cuda-sa`), multi-spin coded
simulated annealing (`quip-cuda-msa`) and heat-bath Gibbs (`quip-cuda-gibbs`),
shipped as separate binaries. **amd64 only.**

Each process binds one CUDA device (`--device N`) and drives it directly.
The coordinator takes N from the `[cuda.N]` config section and passes it as
`--device N`. The miner warns at startup when its `cuda-N` label and `--device`
disagree, and the label never overrides the flag.
Kernels (`kernels/sa.cu`, `kernels/msa.cu`, `kernels/gibbs.cu`) are JIT-compiled via NVRTC at
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
`quip-cuda-msa` sizes its spin state against the opt-in shared memory per
block. A device with at least 96 KB holds two replica words per spin and serves
128 reads. Turing (sm_75, 64 KB) holds one word and serves 64 reads. The miner
declares the read count it can serve when it starts.

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
The model scale is the greatest common divisor, or `GCD`, of its nonzero coefficients in milli units.
At each spin, add `|h|` and all `|J|` on its edges.
The local field bound is the largest of these sums.

- **SA:** solves every whole-milli model up to a local field of 536,870,911 milli.
  SA uses the int8 kernel when `max_i(|h_i| + sum_j |J_ij|) / GCD_step <= 63`.
  This bound keeps the flip delta within int8 because `2 * field <= 127`.
  Other models use the fixed-point wide kernel.
  A complete graph with unit couplings and zero fields has field 63 at 64 nodes and uses int8.
  At 65 nodes, its field is 64 and it uses wide.
  Its accept decision uses integers only, so the same seed gives the same decision on every GPU.
  The SA identity advertises 536,870,910 milli because `quip-solver-core` 0.0.3 `--solve` does not round coefficients (QUI-1813).
  The wide kernel cannot accept a move whose Metropolis probability is below about 7.6e-6.
- **Gibbs:** accepts models whose absolute coefficients are at most 127 times the model's GCD step.
  Gibbs refuses models that need a wide path.
- **MSA:** accepts models of degree at most 20.
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
memory. The device computes no energies. The host rescores every sample with
`energy_milli`, so results are consensus-scored like the other kernels.

The unit kernel uses couplings and fields by sign. Consensus problems have
`J` in {-1, 1} and `h` in {-1, 0, 1}, which it represents exactly.
Other supported whole-milli models use the weighted kernel at their own scale.
The miner refuses any model that exceeds the coefficient or local field bounds.
A spin with more than 20 neighbors cannot run on either MSA kernel.
The miner rejects such a job as `TooLarge`.

Node capacity comes from the device. The spin state uses `dynamic shared memory`,
so the ceiling is the opt-in shared memory per block less 8720 bytes, divided
by 8 times the replica words. At two words that is 5791 spins on a 99 KB part
and 5599 on Volta. `--max-nodes` greater than the ceiling is an error at open, never a
silent clamp.

Replica words fall back from two to one when the device cannot hold the
resolved capacity at two. The read count halves with them, because each 64-bit
word carries 64 replicas. Turing is the case this serves. A 4577-spin topology
needs 73232 bytes at two words and 36616 at one, under a 64 KB ceiling. The
miner then pins reads at the resolved count, so no job asks for more than the
device holds.

Two host-side changes came with it and apply to `quip-cuda-sa` too:

* **Stop on cancel.** When the coordinator cancels a round, in-flight models
  stop at the next rung instead of running to completion. The new round
  starts within milliseconds rather than after a full batch.
* **Parallel scoring.** The host rescores downloaded samples on `QUIP_SCORE_THREADS`
  threads (default 4). With one thread the sampler capped a fast kernel at
  about 20 jobs per second and left the GPU idle between batches.

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

No slot completes. The host waits for a slot to complete while the kernel waits
for a slot to become ready. Any divergence between the two parks both sides, and
each side behaves as written, so nothing below them can detect it. The driver
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
