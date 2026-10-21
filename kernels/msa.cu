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
// implemented by quip-solver-cpu's sa_msc.rs. With b_i the bit of spin i and
// c_ij the bit of J_ij, l_ij = c_ij ^ b_i ^ b_j is set on the replicas where
// the bond is satisfied. Summing the d incident bonds into bit planes gives L
// per replica, and the Metropolis test is the integer comparison
// L <= (d + M) / 2, where M is a geometric draw shared by every replica of a
// word, read from a per-rung threshold row at a random per-sweep offset. A
// field h_i in {-1, 0, +1} is one more bond to a spin pinned at +1.
//
// Preconditions the host enforces (streaming::build_msa_state): every spin
// has at most MSA_MAX_DEG (62) neighbours, counted MSA_CHUNK at a time. The
// unit build reads couplings and fields by sign, and the host sends it unit
// models only. Weighted builds read them at full width, and the host bounds
// the local field at 2^24 - 1 scale units.
//
// Spins are updated one colour class at a time (the host's greedy colouring,
// as gibbs.cu uses), so no two spins updated together interact. Each thread
// owns one word of one spin per step: thread t handles word t % words of
// spins t / words, t / words + blockDim.x / words, and so on.
//
// Spin state: spins 0..k_shared-1 live in dynamic shared memory, spins
// k_shared..N-1 in this nonce's slice of a global spill buffer. The host
// sets k_shared to the most spins the opt-in shared memory holds
// (streaming::msa_layout), so a graph that fits never touches the spill
// buffer. Placement does not change the RNG stream or the update order: a
// spilled run returns the same samples as an all-shared run of the same seed.
// Energies are not computed on the device; the host rescores every sample
// with energy_milli.

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
// field bond need a count up to 63; six planes hold exactly that.
#define MSA_PLANES    6
#define MSA_MAX_COUNT 63
// Acceptance thresholds per rung: cut[m] for m in 0..=MSA_MAX_FIELD.
#define MSA_MAX_FIELD 63
// Threshold row per rung, in bytes. A power of two so the per-sweep offset
// is a mask.
#define MSA_ROW       8192
#define MSA_ROW_MASK  (MSA_ROW - 1)
// Neighbours per counting pass: one popcount21 tree takes MSA_CHUNK
// neighbours plus the field bond. A spin of degree <= MSA_CHUNK takes one
// pass with the inputs the single-pass kernel used.
#define MSA_CHUNK     20
// Most neighbours per spin. With the field bond the count reaches 63, the
// most MSA_PLANES planes hold. The host refuses a denser topology
// (capacity::MSA_MAX_DEGREE mirrors this).
#define MSA_MAX_DEG   62

// Coefficient width. The default build is the unit-weight kernel every
// mining job runs, and it contains no weighted code.
// -DQUIP_MSA_WEIGHTED builds the weighted kernel (QUI-1641, after the CPU
// design in QUI-1398): i32 coefficients and a per-lane local field. Exactly
// one lane arithmetic is chosen: QUIP_MSA_LANE_I32 (integer-only Metropolis
// against the host's -ln(u) table, the same on every GPU) or
// QUIP_MSA_LANE_F32 (an __expf test).
#ifdef QUIP_MSA_WEIGHTED
typedef int msa_coef_t;
#if defined(QUIP_MSA_LANE_I32) == defined(QUIP_MSA_LANE_F32)
#error "a weighted msa build needs exactly one of QUIP_MSA_LANE_I32 and QUIP_MSA_LANE_F32"
#endif
#define NEGLOG_MASK 65535
static_assert(sizeof(msa_coef_t) == 4, "weighted msa uses i32 coefficients");
#else
typedef signed char msa_coef_t;
#endif

typedef unsigned long long u64;

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

// acc += add per lane, MSA_PLANES-bit ripple add. The host bounds every
// count at MSA_MAX_COUNT, so the final carry is always zero.
__device__ __forceinline__ void add_planes(u64* acc, const u64* add) {
    u64 carry = 0ull;
    #pragma unroll
    for (int k = 0; k < MSA_PLANES; ++k) {
        u64 a = acc[k];
        u64 b = add[k];
        u64 t = a ^ b;
        acc[k] = t ^ carry;
        carry = (a & b) | (t & carry);
    }
}

// ==============================================================================
// Spin storage
// ==============================================================================
// Where spin `var`'s replica word `w` lives. SPILL == false is the all-shared
// case (k == N): every access is a shared-memory access, as before the spill
// buffer existed. Spill words use plain loads and stores, never __ldg: this
// block writes them, and __syncthreads makes the writes visible to the
// block's other threads.
template <bool SPILL>
struct SpinStore {
    u64* shared_state;
    u64* spill_state;
    int k;
    int words;

    __device__ __forceinline__ u64& at(int var, int w) const {
        if (!SPILL || var < k) {
            return shared_state[var * words + w];
        }
        return spill_state[(size_t)(var - k) * (size_t)words + w];
    }
};

// One model's inputs and outputs, already offset to the active slot, so both
// SpinStore instantiations of anneal_model share one argument list.
struct MsaModel {
    const int* csr_row_ptr;
    const int* csr_col_ind;
    const int* color_starts;
    const int* color_counts;
    const int* color_nodes;
    int num_colors;
    const msa_coef_t* J;
    const msa_coef_t* h;
    signed char* samples;
    int* energies;
    const float* beta_schedule;
    int num_betas;
    int sweeps_per_beta;
    volatile int* exit_flag;
    int num_reads;
    int N;
    int max_packed_size;
    int words;
    u64 slot_seed;
#ifdef QUIP_MSA_LANE_I32
    const long long* beta_fixed;
    const int* delta_cut;
    const long long* neglog_table;
#endif
};

#ifndef QUIP_MSA_WEIGHTED
// Satisfied-bond count of spin `var` across the 64 lanes of word `w`, into
// MSA_PLANES bit planes, and the bond count d (field included). The field
// rides in the first pass's 21st input; later passes leave it zero.
template <bool SPILL>
__device__ __forceinline__ int count_satisfied(
    const MsaModel& mdl, const SpinStore<SPILL>& st,
    u64 bi, int w, int pstart, int pend, int h, u64* planes) {
    int d = 0;
    bool first = true;
    for (int base = pstart; first || base < pend; base += MSA_CHUNK) {
        int nb[MSA_CHUNK];
        int jj[MSA_CHUNK];
        #pragma unroll
        for (int q = 0; q < MSA_CHUNK; ++q) {
            int p = base + q;
            bool ok = p < pend;
            nb[q] = ok ? __ldg(&mdl.csr_col_ind[p]) : 0;
            jj[q] = ok ? (int)__ldg(&mdl.J[p]) : 0;
        }
        u64 x[MSA_CHUNK + 1];
        #pragma unroll
        for (int q = 0; q < MSA_CHUNK; ++q) {
            int J = jj[q];
            u64 sj = st.at(nb[q], w);
            u64 l = ((J < 0) ? ~0ull : 0ull) ^ bi ^ sj;
            x[q] = (J != 0) ? l : 0ull;
            d += (J != 0);
        }
        if (first) {
            x[MSA_CHUNK] = (h != 0) ? ((h < 0) ? ~bi : bi) : 0ull;
            d += (h != 0);
            popcount21(x, planes);
            first = false;
        } else {
            x[MSA_CHUNK] = 0ull;
            u64 part[MSA_PLANES];
            popcount21(x, part);
            add_planes(planes, part);
        }
    }
    return d;
}
#else
// Per-lane Metropolis test of the weighted kernel, with this spin's draw.
struct LaneTest {
#ifdef QUIP_MSA_LANE_I32
    long long rung_beta;
    int rung_cut;
    long long bound;
#else
    float beta;
    float u;
#endif

    __device__ __forceinline__ bool ok(int field, u64 bi, int lane) const {
        const int de = ((bi >> lane) & 1ull) ? 2 * field : -2 * field;
#ifdef QUIP_MSA_LANE_I32
        return de <= 0 || (de <= rung_cut && (long long)de * rung_beta < bound);
#else
        return de <= 0 || u < __expf(-beta * (float)de);
#endif
    }
};

// Accept mask of spin `var`, word `w`. A spin of degree <= MSA_CHUNK keeps
// its neighbour words in registers, as the single-pass kernel did. A denser
// spin sums per-lane fields chunk by chunk into a local array; integer sums
// are exact in any order, so the result does not depend on the chunking.
template <bool SPILL>
__device__ __forceinline__ u64 weighted_accept(
    const MsaModel& mdl, const SpinStore<SPILL>& st,
    u64 bi, int w, int pstart, int pend, int h, const LaneTest& test) {
    u64 accept = 0ull;
    if (pend - pstart <= MSA_CHUNK) {
        int jj[MSA_CHUNK];
        u64 sw[MSA_CHUNK];
        #pragma unroll
        for (int q = 0; q < MSA_CHUNK; ++q) {
            int p = pstart + q;
            bool ok = p < pend;
            int nb = ok ? __ldg(&mdl.csr_col_ind[p]) : 0;
            jj[q] = ok ? (int)__ldg(&mdl.J[p]) : 0;
            sw[q] = st.at(nb, w);
        }
        for (int lane = 0; lane < 64; ++lane) {
            int field = h;
            #pragma unroll
            for (int q = 0; q < MSA_CHUNK; ++q) {
                // Spin bit 1 is -1. Missing neighbours have J = 0.
                field += ((sw[q] >> lane) & 1ull) ? -jj[q] : jj[q];
            }
            accept |= (u64)test.ok(field, bi, lane) << lane;
        }
        return accept;
    }
    int field[64];
    for (int lane = 0; lane < 64; ++lane) {
        field[lane] = h;
    }
    for (int base = pstart; base < pend; base += MSA_CHUNK) {
        int jj[MSA_CHUNK];
        u64 sw[MSA_CHUNK];
        #pragma unroll
        for (int q = 0; q < MSA_CHUNK; ++q) {
            int p = base + q;
            bool ok = p < pend;
            int nb = ok ? __ldg(&mdl.csr_col_ind[p]) : 0;
            jj[q] = ok ? (int)__ldg(&mdl.J[p]) : 0;
            sw[q] = st.at(nb, w);
        }
        for (int lane = 0; lane < 64; ++lane) {
            int sum = 0;
            #pragma unroll
            for (int q = 0; q < MSA_CHUNK; ++q) {
                sum += ((sw[q] >> lane) & 1ull) ? -jj[q] : jj[q];
            }
            field[lane] += sum;
        }
    }
    for (int lane = 0; lane < 64; ++lane) {
        accept |= (u64)test.ok(field[lane], bi, lane) << lane;
    }
    return accept;
}
#endif

// Anneal one model and pack its reads. Returns true when the host aborted the
// model (EXIT_ABORT); the caller then never publishes the slot. `cut`, `row`
// and `s_abort` are the block's shared scratch.
template <bool SPILL>
__device__ __forceinline__ bool anneal_model(
    const MsaModel& mdl, const SpinStore<SPILL>& st,
    u64* cut, unsigned char* row, int* s_abort, u64 rng) {
    const int tid = threadIdx.x;
    const int words = mdl.words;
    // `words` is a power of two (1 or 2), so the word index is a mask.
    const int w = tid & (words - 1);
    const int g = tid / words;
    const int gstride = blockDim.x / words;
    const int total = mdl.N * words;
    const int packed_size = (mdl.N + 7) / 8;

    // Random initial state: every replica of every spin. Index i is spin
    // i / words, word i % words: the order the all-shared kernel wrote.
    for (int i = tid; i < total; i += blockDim.x) {
        st.at(i / words, i & (words - 1)) = xs64(rng);
    }
    __syncthreads();

    bool aborted = false;
    for (int beta_idx = 0; beta_idx < mdl.num_betas; ++beta_idx) {
        // Abort-on-cancel, decided through shared memory so every thread
        // leaves at the same rung (the loop body has barriers).
        if ((beta_idx & 7) == 0) {
            if (tid == 0) {
                *s_abort = (*mdl.exit_flag == EXIT_ABORT);
            }
            __syncthreads();
            if (*s_abort) {
                aborted = true;
                break;
            }
        }

        float beta = __ldg(&mdl.beta_schedule[beta_idx]);
#ifndef QUIP_MSA_WEIGHTED
        // Acceptance cuts for this rung: cut[m] = exp(-2 beta m) scaled to
        // the u64 range, non-increasing in m.
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
#endif
#ifdef QUIP_MSA_LANE_I32
        const long long rung_beta = __ldg(&mdl.beta_fixed[beta_idx]);
        const int rung_cut = __ldg(&mdl.delta_cut[beta_idx]);
#endif

        for (int sweep = 0; sweep < mdl.sweeps_per_beta; ++sweep) {
            // Random cyclic shift of the row per sweep, uniform across the block.
            int off = (int)(splitmix64(
                mdl.slot_seed ^ ((u64)beta_idx << 20) ^ (u64)sweep) & MSA_ROW_MASK);

            for (int c = 0; c < mdl.num_colors; ++c) {
                int start = __ldg(&mdl.color_starts[c]);
                int cnt = __ldg(&mdl.color_counts[c]);
                for (int k = g; k < cnt; k += gstride) {
                    int var = __ldg(&mdl.color_nodes[start + k]);
                    int pstart = __ldg(&mdl.csr_row_ptr[var]);
                    int pend = __ldg(&mdl.csr_row_ptr[var + 1]);
                    int h = __ldg(&mdl.h[var]);

#ifdef QUIP_MSA_WEIGHTED
                    u64 bi = st.at(var, w);
                    const u64 draw = splitmix64(
                        mdl.slot_seed ^ ((u64)beta_idx << 40)
                        ^ ((u64)sweep << 20) ^ (u64)var);
                    LaneTest test;
#ifdef QUIP_MSA_LANE_I32
                    test.rung_beta = rung_beta;
                    test.rung_cut = rung_cut;
                    test.bound = __ldg(&mdl.neglog_table[draw & NEGLOG_MASK]);
#else
                    test.beta = beta;
                    // 23 bits keep k + 0.5f exact, so u never reaches 1.0f.
                    test.u = ((float)(draw >> 41) + 0.5f) * 1.1920929e-7f;
#endif
                    st.at(var, w) = bi ^ weighted_accept(mdl, st, bi, w, pstart, pend, h, test);
#else
                    u64 bi = st.at(var, w);
                    u64 planes[MSA_PLANES];
                    int d = count_satisfied(mdl, st, bi, w, pstart, pend, h, planes);
                    // Metropolis: accept where L <= (d + M) / 2.
                    int m = row[(var + off) & MSA_ROW_MASK];
                    int limit = (d + m) >> 1;
                    u64 accept = (limit >= d) ? ~0ull : le_constant(planes, limit);
                    st.at(var, w) = bi ^ accept;
#endif
                }
                __syncthreads();
            }
        }
    }

    // Pack replicas 0..num_reads-1 (bit 1 == spin -1, as sa.cu does).
    // Skipped for an aborted model: the host never reads a slot it did not
    // see COMPLETE.
    if (!aborted) {
        for (int r = tid; r < mdl.num_reads; r += blockDim.x) {
            int rw = r >> 6;
            if (rw >= words) rw = words - 1;
            int lane = r & 63;
            signed char* out = &mdl.samples[(long long)r * mdl.max_packed_size];
            for (int b = 0; b < packed_size; ++b) {
                unsigned int byte = 0;
                int base = b * 8;
                #pragma unroll
                for (int bit = 0; bit < 8; ++bit) {
                    int var = base + bit;
                    if (var < mdl.N) {
                        byte |= (unsigned int)((st.at(var, rw) >> lane) & 1ull) << bit;
                    }
                }
                out[b] = (signed char)byte;
            }
            // The host rescores from spins; the kernel tracks no energy.
            mdl.energies[r] = 0;
        }
    }
    return aborted;
}

// ==============================================================================
// KERNEL: Self-feeding persistent multi-spin SA with 3-slot rotating buffers
// ==============================================================================
// Each nonce owns 1 block = 1 SM. Thread 0 manages slot transitions via
// atomicCAS. No inter-block sync needed.
//
// Dynamic shared memory layout (bytes): k_shared * words u64 of spin state,
// then MSA_MAX_FIELD + 1 u64 acceptance cuts, then MSA_ROW bytes of
// threshold draws. The host computes the total (capacity::msa_shared_bytes
// at k_shared) and passes it as the launch's shared_mem_bytes. spill_state
// holds (N - k_shared) * words u64 per nonce, nonce-major.

extern "C" __global__ void cuda_msa_self_feeding(
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,
    const int* __restrict__ color_starts,
    const int* __restrict__ color_counts,
    const int* __restrict__ color_nodes,
    int num_colors,
    const msa_coef_t* __restrict__ slot_J_vals,
    const msa_coef_t* __restrict__ slot_h_vals,
    signed char* slot_samples,
    int* slot_energies,
    const float* __restrict__ beta_schedule,
    int num_betas,
    int sweeps_per_beta,
    volatile int* nonce_ctrl,
    int num_nonces,
    int num_reads,
    int N,
    int nnz,
    int max_packed_size,
    unsigned int base_seed,
    int words,
    u64* spill_state,
    int k_shared
#ifdef QUIP_MSA_LANE_I32
    , const long long* __restrict__ beta_fixed
    , const int* __restrict__ delta_cut
    , const long long* __restrict__ neglog_table
#endif
) {
    extern __shared__ u64 smem[];
    u64* state = smem;                                             // k_shared * words
    u64* cut = smem + (size_t)k_shared * (size_t)words;            // MSA_MAX_FIELD + 1
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

    u64* my_spill = spill_state
        + (size_t)nonce_id * (size_t)(N - (k_shared < N ? k_shared : N)) * (size_t)words;
    const SpinStore<false> all_shared = {state, my_spill, N, words};
    const SpinStore<true> split = {state, my_spill, k_shared, words};

    // === Model loop: process slots until none READY ===
    while (true) {
        int slot_idx = nonce_id * 3 + active_slot;
        MsaModel mdl;
        mdl.csr_row_ptr = csr_row_ptr;
        mdl.csr_col_ind = csr_col_ind;
        mdl.color_starts = color_starts;
        mdl.color_counts = color_counts;
        mdl.color_nodes = color_nodes;
        mdl.num_colors = num_colors;
        mdl.J = &slot_J_vals[(long long)slot_idx * nnz];
        mdl.h = &slot_h_vals[(long long)slot_idx * N];
        mdl.samples = &slot_samples[(long long)slot_idx * num_reads * max_packed_size];
        mdl.energies = &slot_energies[(long long)slot_idx * num_reads];
        mdl.beta_schedule = beta_schedule;
        mdl.num_betas = num_betas;
        mdl.sweeps_per_beta = sweeps_per_beta;
        mdl.exit_flag = &nonce_ctrl[ctrl_base + CTRL_EXIT_NOW];
        mdl.num_reads = num_reads;
        mdl.N = N;
        mdl.max_packed_size = max_packed_size;
        mdl.words = words;
        // RNG: one xorshift64 stream per thread, derived from the slot seed.
        mdl.slot_seed = splitmix64(
            ((u64)base_seed << 8) ^ (u64)(nonce_id * 3 + active_slot));
#ifdef QUIP_MSA_LANE_I32
        mdl.beta_fixed = beta_fixed;
        mdl.delta_cut = delta_cut;
        mdl.neglog_table = neglog_table;
#endif
        u64 rng = splitmix64(mdl.slot_seed ^ ((u64)(tid + 1) * 0x9E3779B97F4A7C15ull));
        if (rng == 0) rng = 0xdeadbeefcafef00dull;

        const bool aborted = (k_shared >= N)
            ? anneal_model<false>(mdl, all_shared, cut, row, &s_abort, rng)
            : anneal_model<true>(mdl, split, cut, row, &s_abort, rng);

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
