// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

// ==============================================================================
// CUDA MULTI-SPIN CODED SIMULATED ANNEALING - SELF-FEEDING PERSISTENT KERNEL
// ==============================================================================
// Architecture: self-feeding persistent kernel with 3-slot rotating buffers,
// the same control plane and output layout as cuda_sa_self_feeding in sa.cu,
// so the host session code is shared. Each nonce owns exactly 1 block (1 SM).
// One block anneals one model: 64 replicas share each 64-bit word of spin
// state, `words` words per spin, so a 128-read job is two words.
//
// Algorithm: Isakov, Zintchenko, Ronnow, Troyer, "Optimised simulated
// annealing for Ising spin glasses", Comput. Phys. Commun. 192 (2015), as
// implemented by quip-miner-cpu's sa_msc.rs. With b_i the bit of spin i and
// c_ij the bit of J_ij, l_ij = c_ij ^ b_i ^ b_j is set on the replicas where
// the bond is satisfied. Summing the d incident bonds into bit planes gives L
// per replica, and the Metropolis test is the integer comparison
// L <= (d + M) / 2, where M is a geometric draw shared by every replica of a
// word, read from a per-rung threshold row at a random per-sweep offset. A
// field h_i in {-1, 0, +1} is one more bond to a spin pinned at +1.
//
// Preconditions the host enforces (streaming::build_msa_state): every spin
// has at most MSA_MAX_DEG neighbours. Couplings and fields are used by sign,
// so |J| > 1 or |h| > 1 degrade the anneal but never the host-side scoring,
// exactly as the int8 quantisation in sa.cu does.
//
// Spins are updated one colour class at a time (the host's greedy colouring,
// as gibbs.cu uses), so no two spins updated together interact. Each thread
// owns one word of one spin per step: thread t handles word t % words of
// spins t / words, t / words + blockDim.x / words, and so on.
//
// Spin state lives in dynamic shared memory sized per launch, so there is no
// QUIP_MAX_NODES here: the host's capacity model (capacity::msa_budget) bounds
// N against the device's opt-in shared memory. Energies are not computed on
// the device; the host rescores every sample with energy_milli.

// ==============================================================================
// NonceControl layout (flat int array, CTRL_STRIDE ints per nonce)
// ==============================================================================
// Slot states
#define SLOT_EMPTY    0
#define SLOT_READY    1
#define SLOT_ACTIVE   2
#define SLOT_COMPLETE 3

// Field offsets within each nonce's control block
#define CTRL_STRIDE       8
#define CTRL_SLOT_STATE_0 0
#define CTRL_SLOT_STATE_1 1
#define CTRL_SLOT_STATE_2 2
#define CTRL_ACTIVE_SLOT  3
// [4] unused (blocks_done not needed for 1 block/nonce)
// [5] unused (work_queue not needed for 1 block/nonce)
#define CTRL_EXIT_NOW     6
// [7] unused (generation not needed for 1 block/nonce)

// Values the host writes to CTRL_EXIT_NOW. Any nonzero value ends the waits
// for a READY slot. Only EXIT_ABORT is honoured inside the sweep loop: the
// block leaves mid-model and never publishes the slot. EXIT_AFTER_MODEL lets
// the current model finish and publish first, which the isolated bench path
// (streaming::bench_one) relies on when it pre-arms the flag.
#define EXIT_AFTER_MODEL 1
#define EXIT_ABORT       2

// ==============================================================================
// Multi-spin constants
// ==============================================================================
// Bit planes of the satisfied-bond counter. MSA_MAX_DEG neighbours plus one
// field bond need a count up to 21; six planes hold up to 63.
#define MSA_PLANES    6
#define MSA_MAX_COUNT 63
// Acceptance thresholds per rung: cut[m] for m in 0..=MSA_MAX_FIELD.
#define MSA_MAX_FIELD 63
// Threshold row per rung, in bytes. A power of two so the per-sweep offset
// is a mask.
#define MSA_ROW       8192
#define MSA_ROW_MASK  (MSA_ROW - 1)
// Neighbours prefetched and counted per spin: Zephyr's degree bound. The
// host refuses a denser topology (capacity::MSA_MAX_DEGREE mirrors this).
#define MSA_MAX_DEG   20

typedef unsigned long long u64;

extern "C" {

// xorshift64: one stream per thread, seeded from the slot seed below.
__device__ __forceinline__ u64 xs64(u64 &s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

// splitmix64 finaliser, used to derive per-slot, per-thread and per-sweep
// seeds from small integers.
__device__ __forceinline__ u64 splitmix64(u64 x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Carry-save adder: (h, l) = a + b + c per lane.
__device__ __forceinline__ void csa(u64 &h, u64 &l, u64 a, u64 b, u64 c) {
    u64 u = a ^ b;
    h = (a & b) | (u & c);
    l = u ^ c;
}

// Per-lane popcount of 21 one-bit inputs (missing inputs are zero words)
// into planes[0..5] = ones, twos, fours, eights, sixteens, 0. A Harley-Seal
// tree: about 100 operations instead of 21 x 18 for a ripple add.
__device__ __forceinline__ void popcount21(const u64* x, u64* planes) {
    u64 ones = 0, twos = 0, fours = 0, eights = 0;
    u64 tA, tB, fA, fB, eA, eB, sA, sB;
    csa(tA, ones, ones, x[0], x[1]);
    csa(tB, ones, ones, x[2], x[3]);
    csa(fA, twos, twos, tA, tB);
    csa(tA, ones, ones, x[4], x[5]);
    csa(tB, ones, ones, x[6], x[7]);
    csa(fB, twos, twos, tA, tB);
    csa(eA, fours, fours, fA, fB);
    csa(tA, ones, ones, x[8], x[9]);
    csa(tB, ones, ones, x[10], x[11]);
    csa(fA, twos, twos, tA, tB);
    csa(tA, ones, ones, x[12], x[13]);
    csa(tB, ones, ones, x[14], x[15]);
    csa(fB, twos, twos, tA, tB);
    csa(eB, fours, fours, fA, fB);
    csa(sA, eights, eights, eA, eB);
    csa(tA, ones, ones, x[16], x[17]);
    csa(tB, ones, ones, x[18], x[19]);
    csa(fA, twos, twos, tA, tB);
    tA = ones & x[20];
    ones ^= x[20];
    fB = twos & tA;
    twos ^= tA;
    csa(eA, fours, fours, fA, fB);
    sB = eights & eA;
    eights ^= eA;
    planes[0] = ones;
    planes[1] = twos;
    planes[2] = fours;
    planes[3] = eights;
    // At most 21 inputs, so at most one of the two sixteens carries is set.
    planes[4] = sA | sB;
    planes[5] = 0ull;
}

// Mask of lanes whose 6-bit counter is <= limit (bit-serial compare, LSB
// first). Total: a limit at or above MSA_MAX_COUNT admits every count.
__device__ __forceinline__ u64 le_constant(const u64* planes, int limit) {
    int bound = limit + 1;
    if (bound > MSA_MAX_COUNT) {
        return ~0ull;
    }
    u64 ge = ~0ull;
    #pragma unroll
    for (int k = 0; k < MSA_PLANES; ++k) {
        u64 set = 0ull - (u64)((bound >> k) & 1);
        u64 p = planes[k];
        u64 both = p & ge;
        u64 either = p ^ ge;
        ge = both | (either & ~set);
    }
    return ~ge;
}

// ==============================================================================
// KERNEL: Self-feeding persistent multi-spin SA with 3-slot rotating buffers
// ==============================================================================
// Each nonce owns 1 block = 1 SM. The block anneals every replica word of
// every spin; thread 0 manages slot transitions via atomicCAS. No
// inter-block sync needed.
//
// Dynamic shared memory layout (bytes): N * words u64 of spin state, then
// MSA_MAX_FIELD + 1 u64 acceptance cuts, then MSA_ROW bytes of threshold
// draws. The host computes the total (capacity::msa_shared_bytes) and passes
// it as the launch's shared_mem_bytes.

__global__ void cuda_msa_self_feeding(
    // Shared topology (constant across all slots)
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,

    // Colour classes (shared topology, not tiled per nonce)
    const int* __restrict__ color_starts,
    const int* __restrict__ color_counts,
    const int* __restrict__ color_nodes,
    int num_colors,

    // Per-slot flat buffers (slot_idx = nonce_id * 3 + slot_id)
    const signed char* __restrict__ slot_J_vals,
    const signed char* __restrict__ slot_h_vals,
    signed char* slot_samples,
    int* slot_energies,

    // Shared beta schedule
    const float* __restrict__ beta_schedule,
    int num_betas,
    int sweeps_per_beta,

    // Control array (CTRL_STRIDE ints per nonce)
    volatile int* nonce_ctrl,

    // Config
    int num_nonces,
    int num_reads,
    int N,
    int nnz,
    int max_packed_size,
    unsigned int base_seed,
    int words
) {
    extern __shared__ u64 smem[];
    u64* state = smem;                                        // N * words
    u64* cut = smem + (size_t)N * (size_t)words;              // MSA_MAX_FIELD + 1
    unsigned char* row = (unsigned char*)(cut + (MSA_MAX_FIELD + 1));  // MSA_ROW
    // Static members. capacity::MSA_STATIC_SHARED_BYTES (16) bounds their
    // size: the host refuses a loaded function that reports more.
    __shared__ int s_active_slot;
    __shared__ int s_abort;

    int tid = threadIdx.x;
    int nonce_id = blockIdx.x;
    if (nonce_id >= num_nonces) return;

    int ctrl_base = nonce_id * CTRL_STRIDE;

    // Thread 0: claim first READY slot via atomicCAS. Waits for a slot
    // instead of giving up after one pass, for the reasons on the same wait
    // in sa.cu (QUI-828): the host's cold-start uploads run on another
    // stream, so a block can start before its slot is marked READY. Only an
    // explicit host EXIT_NOW ends the wait, and teardown always raises it.
    int active_slot = -1;
    if (tid == 0) {
        while (true) {
            for (int s = 0; s < 3; s++) {
                int old = atomicCAS(
                    (int*)&nonce_ctrl[ctrl_base + s],
                    SLOT_READY, SLOT_ACTIVE
                );
                if (old == SLOT_READY) {
                    active_slot = s;
                    break;
                }
            }
            if (active_slot >= 0) {
                nonce_ctrl[ctrl_base + CTRL_ACTIVE_SLOT] = active_slot;
                __threadfence();
                break;
            }
            if (nonce_ctrl[ctrl_base + CTRL_EXIT_NOW]) {
                break;  // host ended the stream
            }
            __nanosleep(10000);  // 10us
        }
        s_active_slot = active_slot;
    }
    __syncthreads();
    active_slot = s_active_slot;
    if (active_slot < 0) return;

    // Thread-to-word mapping. `words` is a power of two (1, 2 or 4), so the
    // word index is a mask of the thread id.
    const int w = tid & (words - 1);
    const int g = tid / words;
    const int gstride = blockDim.x / words;
    const int total = N * words;
    const int packed_size = (N + 7) / 8;

    // === Model loop: process slots until none READY ===
    while (true) {
        int slot_idx = nonce_id * 3 + active_slot;
        const signed char* my_J = &slot_J_vals[(long long)slot_idx * nnz];
        const signed char* my_h = &slot_h_vals[(long long)slot_idx * N];
        long long sample_base = (long long)slot_idx * num_reads * max_packed_size;
        long long energy_base = (long long)slot_idx * num_reads;

        // RNG: one xorshift64 stream per thread, derived from the slot seed.
        u64 slot_seed = splitmix64(
            ((u64)base_seed << 8) ^ (u64)(nonce_id * 3 + active_slot));
        u64 rng = splitmix64(slot_seed ^ ((u64)(tid + 1) * 0x9E3779B97F4A7C15ull));
        if (rng == 0) rng = 0xdeadbeefcafef00dull;

        // Random initial state: every replica of every spin.
        for (int i = tid; i < total; i += blockDim.x) {
            state[i] = xs64(rng);
        }
        __syncthreads();

        // === Multi-spin sweep loop ===
        bool aborted = false;
        for (int beta_idx = 0; beta_idx < num_betas; ++beta_idx) {
            // Abort-on-cancel: the host writes EXIT_ABORT when the
            // coordinator abandons the round. The decision goes through
            // shared memory so every thread leaves the loop at the same
            // rung; the loop body has barriers, so a per-thread break would
            // deadlock the block.
            if ((beta_idx & 7) == 0) {
                if (tid == 0) {
                    s_abort = (nonce_ctrl[ctrl_base + CTRL_EXIT_NOW] == EXIT_ABORT);
                }
                __syncthreads();
                if (s_abort) {
                    aborted = true;
                    break;
                }
            }

            // Acceptance cuts for this rung: cut[m] = exp(-2 beta m) scaled
            // to the u64 range, non-increasing in m.
            float beta = __ldg(&beta_schedule[beta_idx]);
            if (tid <= MSA_MAX_FIELD) {
                double p = exp(-2.0 * (double)beta * (double)tid);
                cut[tid] = (p >= 1.0) ? ~0ull : (u64)(p * 18446744073709551616.0);
            }
            __syncthreads();

            // Threshold row for this rung: each entry is the largest m with
            // u < cut[m], the geometric draw the paper shares across lanes.
            for (int i = tid; i < MSA_ROW; i += blockDim.x) {
                u64 u = xs64(rng);
                int m = 0;
                if (u < cut[1]) {
                    m = 1;
                    while (m < MSA_MAX_FIELD && u < cut[m + 1]) {
                        ++m;
                    }
                }
                row[i] = (unsigned char)m;
            }
            __syncthreads();

            for (int sweep = 0; sweep < sweeps_per_beta; ++sweep) {
                // Random cyclic shift of the row per sweep, uniform across
                // the block.
                int off = (int)(splitmix64(
                    slot_seed ^ ((u64)beta_idx << 20) ^ (u64)sweep) & MSA_ROW_MASK);

                for (int c = 0; c < num_colors; ++c) {
                    int start = __ldg(&color_starts[c]);
                    int cnt = __ldg(&color_counts[c]);
                    for (int k = g; k < cnt; k += gstride) {
                        int var = __ldg(&color_nodes[start + k]);
                        int pstart = __ldg(&csr_row_ptr[var]);
                        int pend = __ldg(&csr_row_ptr[var + 1]);
                        int h = __ldg(&my_h[var]);

                        // Prefetch every neighbour index and coupling
                        // (independent global loads, L1-resident) before
                        // touching shared memory.
                        int nb[MSA_MAX_DEG];
                        int jj[MSA_MAX_DEG];
                        #pragma unroll
                        for (int q = 0; q < MSA_MAX_DEG; ++q) {
                            int p = pstart + q;
                            bool ok = p < pend;
                            nb[q] = ok ? __ldg(&csr_col_ind[p]) : 0;
                            jj[q] = ok ? (int)__ldg(&my_J[p]) : 0;
                        }

                        // Satisfied-bond bits per neighbour, plus the field
                        // as a ghost bond to a spin pinned at +1.
                        u64 bi = state[var * words + w];
                        u64 x[MSA_MAX_DEG + 1];
                        int d = 0;
                        #pragma unroll
                        for (int q = 0; q < MSA_MAX_DEG; ++q) {
                            int J = jj[q];
                            u64 sj = state[nb[q] * words + w];
                            u64 l = ((J < 0) ? ~0ull : 0ull) ^ bi ^ sj;
                            x[q] = (J != 0) ? l : 0ull;
                            d += (J != 0);
                        }
                        x[MSA_MAX_DEG] = (h != 0) ? ((h < 0) ? ~bi : bi) : 0ull;
                        d += (h != 0);

                        // Metropolis: accept where L <= (d + M) / 2.
                        u64 planes[MSA_PLANES];
                        popcount21(x, planes);
                        int m = row[(var + off) & MSA_ROW_MASK];
                        int limit = (d + m) >> 1;
                        u64 accept = (limit >= d) ? ~0ull : le_constant(planes, limit);
                        state[var * words + w] = bi ^ accept;
                    }
                    __syncthreads();
                }
            }
        }

        // Pack replicas 0..num_reads-1 (bit 1 == spin -1, as sa.cu does).
        // Skipped for an aborted model: the host never reads a slot it did
        // not see COMPLETE.
        if (!aborted) {
            for (int r = tid; r < num_reads; r += blockDim.x) {
                int rw = r >> 6;
                if (rw >= words) rw = words - 1;
                int lane = r & 63;
                signed char* out =
                    &slot_samples[sample_base + (long long)r * max_packed_size];
                for (int b = 0; b < packed_size; ++b) {
                    unsigned int byte = 0;
                    int base = b * 8;
                    #pragma unroll
                    for (int bit = 0; bit < 8; ++bit) {
                        int var = base + bit;
                        if (var < N) {
                            byte |= (unsigned int)((state[var * words + rw] >> lane) & 1ull)
                                    << bit;
                        }
                    }
                    out[b] = (signed char)byte;
                }
                // The host rescores from spins; the kernel tracks no energy.
                slot_energies[energy_base + r] = 0;
            }
        }

        // All threads done writing samples
        __syncthreads();

        // Fence every thread's global writes so the host sees the samples
        // before SLOT_COMPLETE. __syncthreads is a thread barrier, NOT a
        // memory fence (see sa.cu).
        __threadfence();
        __syncthreads();

        // Thread 0: publish, then find the next READY slot
        if (tid == 0) {
            if (aborted) {
                // Never publish a half-annealed model: the host reads only
                // slots it saw COMPLETE.
                s_active_slot = -1;
            } else {
                nonce_ctrl[ctrl_base + active_slot] = SLOT_COMPLETE;

                // Check exit flag
                if (nonce_ctrl[ctrl_base + CTRL_EXIT_NOW]) {
                    s_active_slot = -1;
                } else {
                    // Find next READY slot. Waits through transient feeder
                    // starvation; only an explicit host EXIT_NOW ends it
                    // (see the same wait in sa.cu, QUI-828).
                    int next_slot = -1;
                    while (true) {
                        for (int s = 0; s < 3; s++) {
                            int old = atomicCAS(
                                (int*)&nonce_ctrl[ctrl_base + s],
                                SLOT_READY, SLOT_ACTIVE
                            );
                            if (old == SLOT_READY) {
                                next_slot = s;
                                break;
                            }
                        }
                        if (next_slot >= 0) break;
                        if (nonce_ctrl[ctrl_base + CTRL_EXIT_NOW]) {
                            break;  // host ended the stream
                        }
                        __nanosleep(10000);  // 10us
                    }

                    if (next_slot >= 0) {
                        nonce_ctrl[ctrl_base + CTRL_ACTIVE_SLOT] = next_slot;
                        __threadfence();
                    }
                    s_active_slot = next_slot;
                }
            }
        }
        __syncthreads();
        active_slot = s_active_slot;
        if (active_slot < 0) return;
    }
}

}  // extern "C"
