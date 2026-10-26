# Project rules

This project uses **bd** (beads) for issue tracking. Run `bd prime` for the workflow context.

Issues live in a local Dolt database at `.beads/dolt/`.
Cross-machine sync uses `bd dolt push/pull`, a git-compatible protocol.
It stores data under `refs/dolt/data` on the git remote.
Code lives under `refs/heads/*`. The `.beads/issues.jsonl` file is a passive export.
The wire protocol does not use this file.

See [SYNC_CONCEPTS.md](https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md) for an overview.
Treat the Dolt database as the source of truth. Do not use `bd import` during normal work.
Try the default sync before third-party Dolt hosting.

## Quick reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

## Shell commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Some systems alias `cp`, `mv`, and `rm` to include `-i` for interactive mode. These commands can then wait for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var



## Card support

Supported GPUs: compute capability 7.0 (Volta) through 12.1 (consumer
Blackwell). The floor is a kernel property (all three kernels use `__nanosleep`,
an sm_70+ instruction). The ceiling comes from NVRTC 12.9, which the
`cuda-12090` cudarc pin in `Cargo.toml` selects. The pin decides which
`libnvrtc.so.N` the miner dlopens at runtime, so it moves together with the
`cuda-nvrtc-12-9` package in quip-solver's `docker/Dockerfile.quip-solver-cuda`. `SUPPORTED_ARCHS` in
`src/cuda_device.rs` is the contract. The test in `tests/arch_coverage.rs` compiles every
kernel for every entry (`make test-archs`). Changing card support is a
reviewed edit to that list, never a side effect of a toolkit bump.

`quip-cuda-msa` keeps the first spins in shared memory and spills the rest to
a per-nonce global buffer. The opt-in shared memory per block sets how many
spins stay on chip. It does not set the node ceiling.
The device probe selects 64 reads on Turing and 128 reads with larger shared-memory budgets.
Node and coupler limits come from device memory divided among instances,
one per SM. With 15 GiB free, 48 SMs and 128 reads, the A4000
advertises 3,987,504 uncoupled nodes or 44,486,028 dense couplers at 9,433 complete nodes.
These independent Binary limits use the default 64-beta schedule. Each model
must pass the memory check with its own precision and schedule length.
The miner refuses a session whose buffers exceed device memory. At
200,000 nodes and 4,000,000 edges with 48 models, the A4000 measured about 2 GiB
for unit models and 5.2 GiB for weighted i32 models (see the
[A4000 spill measurements](docs/perf/2026-10-msa-dram-spill-a4000.md)).
