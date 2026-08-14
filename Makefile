# Local + CI entry points. Everything runs inside the CUDA 12.9 CI image so a
# local run and a pipeline run exercise the same toolkit (see the spec:
# quip-miner docs/superpowers/specs/2026-08-14-cuda-version-target-design.md).
CI_IMAGE := quip-miner-cuda-ci
RUN_CI   := docker run --rm -v $(PWD):/src -v quip-cuda-cargo:/root/.cargo/registry \
            -v quip-cuda-target:/src/target -w /src $(CI_IMAGE)
RUN_GPU  := docker run --rm --gpus all -v $(PWD):/src -v quip-cuda-cargo:/root/.cargo/registry \
            -v quip-cuda-target:/src/target -w /src $(CI_IMAGE)

.PHONY: ci-image check test test-archs test-gpu bench-arch

ci-image:
	docker build -f Dockerfile.ci -t $(CI_IMAGE) .

check: ci-image
	$(RUN_CI) sh -c 'cargo fmt --all --check && cargo clippy --release --all-targets --all-features -- -D warnings'

test: ci-image
	$(RUN_CI) cargo test --release

# Arch coverage needs NVRTC 12.9 but no GPU.
test-archs: ci-image
	$(RUN_CI) cargo test --release --test arch_coverage -- --include-ignored

# The #[ignore]d GPU suite, on real hardware.
test-gpu: ci-image
	$(RUN_GPU) cargo test --release -- --include-ignored

# Benchmark for the spec's D4 gate. OUT differs per run: make bench-arch OUT=bench-out/baseline
OUT ?= bench-out/run
bench-arch: ci-image
	mkdir -p $(OUT)
	$(RUN_GPU) sh -c 'cargo run --release --bin quip-cuda-sa -- bench run \
	    --reads 8 --sweeps 1024 --sweeps-per-beta 4 --nodes 512 --repeats 5 --out /src/$(OUT)'
