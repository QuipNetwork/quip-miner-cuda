# quip-miner-cuda

CUDA Ising miners for the [quip.network](https://gitlab.com/quip.network) v0.3
mining protocol: simulated annealing (`quip-cuda-sa`) and heat-bath Gibbs
(`quip-cuda-gibbs`), shipped as separate binaries. **amd64 only.**

Each process binds one CUDA device (`--device N`) and drives it directly.
Kernels (`kernels/sa.cu`, `kernels/gibbs.cu`) are JIT-compiled via NVRTC at
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
| `quip-cuda-gibbs` | heat-bath Gibbs |

Prebuilt `amd64` binaries are attached to each
[Release](https://gitlab.com/quip.network/quip-miner-cuda/-/releases).

## Build

```sh
cargo build --release        # needs protoc on PATH (protobuf-compiler)
```

`cudarc` uses dynamic-loading, so the build links against no CUDA libraries;
the CUDA driver is loaded and kernels are compiled at process start.

Shared protocol crates (`quip-proto`, `quip-protocol`, `quip-miner-core`) are
git dependencies pinned to a `shared-vX.Y.Z` tag of `quip-protocol`.

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

Conformance/golden and handshake tests drive the binary in isolation via
`quip-mock-coordinator` and check energies against `conformance/golden_vectors.json`.
Tests that need a live CUDA device are marked `#[ignore]`, so a machine without a
GPU reports them as ignored rather than passed. Run them with `--include-ignored`
on a CUDA host.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
