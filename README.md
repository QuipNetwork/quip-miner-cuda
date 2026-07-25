# quip-miner-cuda

CUDA Ising miners for the [quip.network](https://gitlab.com/quip.network) v0.3
mining protocol: simulated annealing (`quip-cuda-sa`) and heat-bath Gibbs
(`quip-cuda-gibbs`), shipped as separate binaries. **amd64 only.**

Each process binds one CUDA device (`--device N`) and drives it directly.
Kernels (`kernels/sa.cu`, `kernels/gibbs.cu`) are JIT-compiled via NVRTC at
runtime through `cudarc`'s dynamic-loading feature, so **building this crate
does not require the CUDA toolkit** — only a CUDA GPU and driver are needed to
*run* the binaries. Energies are scored with the canonical
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
