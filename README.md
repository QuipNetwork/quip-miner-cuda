# quip-miner-cuda

CUDA Ising miners for the [quip.network](https://gitlab.com/quip.network) v0.3
mining protocol: simulated annealing (`quip-cuda-sa`), multi-spin coded
simulated annealing (`quip-cuda-msa`) and heat-bath Gibbs (`quip-cuda-gibbs`),
shipped as separate binaries. **amd64 only.**

Each process binds one CUDA device (`--device N`) and drives it directly.
The coordinator takes N from the `[cuda.N]` config section and passes it as
`--device N`; the miner warns at startup when its `cuda-N` label and `--device`
disagree, and the label never overrides the flag.
Kernels (`kernels/sa.cu`, `kernels/msc.cu`, `kernels/gibbs.cu`) are JIT-compiled via NVRTC at
runtime through `cudarc`'s dynamic-loading feature, so **building this crate
does not require the CUDA toolkit** — only a CUDA GPU and driver are needed to
*run* the binaries.

Supported GPUs: compute capability 7.0 (Volta) through 12.1 (consumer
Blackwell). The floor comes from the kernels (`__nanosleep` is sm_70+); the
ceiling from NVRTC 12.9 (`cuda-12090` in `Cargo.toml`). Kernels are compiled
for each device's detected capability, clamped into that range; capabilities
the toolkit lacks (for example 8.8) get the next lower architecture and load
through the driver's forward-compatible PTX JIT. `SUPPORTED_ARCHS` in
`src/cuda_device.rs` is the contract and `tests/arch_coverage.rs` enforces it
(`make test-archs`). Energies are scored with the canonical
`quip_protocol::scoring::energy_milli` so results match consensus.

## Binaries

| binary | algorithm |
|--------|-----------|
| `quip-cuda-sa` | simulated annealing (Metropolis) |
| `quip-cuda-msa` | multi-spin coded simulated annealing (64 replicas per word) |
| `quip-cuda-gibbs` | heat-bath Gibbs |

Prebuilt `amd64` binaries are attached to each
[Release](https://gitlab.com/quip.network/quip-miner-cuda/-/releases).

## Build

```sh
cargo build --release
```

`cudarc` uses dynamic-loading, so the build links against no CUDA libraries;
the CUDA driver is loaded and kernels are compiled at process start.

The solver contract (`quip-proto`, `quip-protocol`, `quip-solver-core`) is
published to crates.io from [quip.network/quip-solver-core](https://gitlab.com/quip.network/quip-solver-core).

## Running

Requires a CUDA-capable GPU and driver at runtime.

**Connect to a coordinator** (production):

```sh
quip-cuda-sa --quip-coordinator unix:///run/quip/coord.sock --device 0
```

**Driver / fixed-input (run in isolation, no chain).** Use the coordinator's
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

`kernels/msc.cu` is a CUDA port of the multi-spin coded simulated annealing
in `quip-miner-cpu`'s `quip-cpu-msa` (Isakov, Zintchenko, Rønnow, Troyer,
*Optimised simulated annealing for Ising spin glasses*, Comput. Phys. Commun.
192, 2015). 64 replicas share one 64-bit word per spin (two words per spin,
128 reads per job), the Metropolis test is an integer comparison against a
per-rung geometric threshold table, and spins are updated one colour class at
a time (the host's greedy colouring, as the Gibbs kernel uses) so a whole
block anneals one problem in shared memory. Energies are not computed on the
device: the host rescores every sample with `energy_milli`, so results are
consensus-scored exactly like the other kernels. It needs integer couplings
and no fields, which the v0.3 problems satisfy; other graphs fall back to the
host's checks at session build.

Measured on an RTX 5090 Laptop (82 SMs) against the same 24 drawn problems:
at 7392 x 128 it matches the CPU multi-spin solver's depth in 2.0 s per model
on one SM, and in production at 29568 x 128 it runs ~32 jobs/s where the
float kernel managed 0.4 jobs/s at 7392 x 220.

Two host-side changes come with it and apply to `quip-cuda-sa` too:

* **Abort on cancel.** When the coordinator cancels a round, in-flight models
  are aborted at the next rung instead of running to completion, so the new
  round starts within milliseconds rather than after a full batch.
* **Parallel scoring.** Downloaded samples are rescored on
  `QUIP_SCORE_THREADS` host threads (default 4). With one thread the sampler
  capped a fast kernel at ~20 jobs/s and left the GPU idle between batches.

Diagnostics: `QUIP_MSC_DIAG=1|2|3` adds `-DMSC_DIAG=N` at JIT time (no
threshold rows, no spin updates, neither) to isolate kernel cost; pair it
with `QUIP_CUDA_CACHE_DISABLE=1` since defines are not part of the cache key.

## Yielding to other GPU users

`--yielding` lets the miner share a GPU. The NVML governor measures the load
from other processes and compares it against `--utilization`. It does not use
device-wide utilization for this decision, because that figure counts the
miner's own kernels. A busy miner holds device-wide utilization near 100
percent, so a governor that read it would throttle against itself.

When another process passes the ceiling, the miner ends its current session and
waits. Ending the session is what frees the SMs. The kernel is persistent and
holds its SMs until teardown, so a pause inside a session yields nothing.

Per-process attribution needs both NVML support and a process ID that matches.
Inside a container, or on WSL2, the governor falls back to device-wide
utilization and logs which method it uses. The fallback applies only while
another process holds a context. A miner alone on a GPU never throttles.

## Session recovery

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
quip-miner-cuda: no slot completed in 612.4s; rebuilding the self-feeding session
```

## Driver time budget

The stream driver can account for its own wall clock, one window at a time.
Use it to find where driver time goes when throughput falls.

Accounting is off by default. Three environment variables control it:

| Variable | Effect |
| -- | -- |
| `QUIP_DRIVER_BUDGET` | Set to `1` to turn accounting on. Any other value leaves it off. |
| `QUIP_DRIVER_BUDGET_WINDOW` | Report period in seconds. Default 60. |
| `QUIP_DRIVER_BUDGET_OUT` | Path to append one JSON object per window. Optional. |

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
records `att/s` next to board power, core clock, and temperature from NVML:

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
Tests that need a live CUDA device are marked `#[ignore]`, so a machine without a
GPU reports them as ignored rather than passed. Run them with `--include-ignored`
on a CUDA host.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
