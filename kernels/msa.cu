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
// k_shared..N-1 in this nonce's slice of a global spill buffer.
// capacity::MSA_MAX_NODES bounds N; shared memory only sets how many spins
// stay on chip. The host sets k_shared to the most spins the opt-in shared
// memory holds (streaming::msa_layout), so a graph that fits never touches
// the spill buffer. Placement does not change the RNG stream or the update
// order: a spilled run returns the same samples as an all-shared run of the
// same seed.
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
// design in QUI-1398): i32 coefficients by default and a per-lane local field.
// QUIP_MSA_COEF_I8 selects i8 storage; QUIP_MSA_COEF_F32 selects real storage. Exactly
// one lane arithmetic is chosen: QUIP_MSA_LANE_I32 (integer-only Metropolis
// against the host's -ln(u) table, the same on every GPU) or
// QUIP_MSA_LANE_F32 (an __expf test).
#ifdef QUIP_MSA_WEIGHTED
#if defined(QUIP_MSA_COEF_I8) && defined(QUIP_MSA_COEF_F32)
#error "choose exactly one weighted coefficient storage type"
#endif
#if defined(QUIP_MSA_COEF_I8)
typedef signed char msa_coef_t;
#elif defined(QUIP_MSA_COEF_F32)
typedef float msa_coef_t;
#ifndef QUIP_MSA_LANE_F32
#error "f32 coefficient storage requires the f32 lane"
#endif
#else
typedef int msa_coef_t;
#endif
#if defined(QUIP_MSA_LANE_I32) == defined(QUIP_MSA_LANE_F32)
#error "a weighted msa build needs exactly one of QUIP_MSA_LANE_I32 and QUIP_MSA_LANE_F32"
#endif
#define NEGLOG_MASK 65535
#ifndef QUIP_MSA_COEF_I8
static_assert(sizeof(msa_coef_t) == 4, "wide coefficient storage is four bytes");
#endif
#else
typedef signed char msa_coef_t;
#endif

// Preserve integer local fields for the existing builds, including the unit lane.
#ifdef QUIP_MSA_COEF_F32
typedef float msa_field_t;
#else
typedef int msa_field_t;
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
// Per-lane Metropolis test of the dense weighted path, with this spin's draw.
// arm() turns the draw into the threshold; weighted_accept_dense calls it once
// per spin. The single-pass path in anneal_model keeps its own copies of this
// code (the draw hash, the threshold and the accept predicate) because a
// shared helper measured +8.6% to +10.5% on weighted-i32 against main.
struct LaneTest {
    u64 draw;
#ifdef QUIP_MSA_LANE_I32
    long long rung_beta;
    int rung_cut;
    const long long* neglog_table;
    long long bound;

    __device__ __forceinline__ void arm() {
        bound = __ldg(&neglog_table[draw & NEGLOG_MASK]);
    }
#else
    float beta;
    float u;

    __device__ __forceinline__ void arm() {
        // 23 bits keep k + 0.5f exact, so u never reaches 1.0f.
        u = ((float)(draw >> 41) + 0.5f) * 1.1920929e-7f;
    }
#endif

    // Same predicate as the inline one in anneal_model's single-pass path.
    __device__ __forceinline__ bool ok(msa_field_t field, u64 bi, int lane) const {
        const msa_field_t de = ((bi >> lane) & 1ull) ? 2 * field : -2 * field;
#ifdef QUIP_MSA_LANE_I32
        return de <= 0 || (de <= rung_cut && (long long)de * rung_beta < bound);
#else
        return de <= 0 || u < __expf(-beta * (float)de);
#endif
    }
};

// Accept mask of a spin denser than MSA_CHUNK. Lanes go in groups of
// MSA_LANE_GROUP, whose partial fields stay in registers while the neighbour
// words are reloaded chunk by chunk for each group. Integer sums are exact in
// any order, so the result does not depend on the chunking. Only spins denser
// than MSA_CHUNK come here; the others take the single-pass code in
// anneal_model, whose neighbour words stay in registers.
#define MSA_LANE_GROUP 8
template <bool SPILL>
__device__ __forceinline__ u64 weighted_accept_dense(
    const int* col_ind, const msa_coef_t* J, const SpinStore<SPILL>& st,
    u64 bi, int w, int pstart, int pend, msa_field_t h, LaneTest test) {
    test.arm();
    u64 accept = 0ull;
    for (int lane0 = 0; lane0 < 64; lane0 += MSA_LANE_GROUP) {
        msa_field_t field[MSA_LANE_GROUP];
        #pragma unroll
        for (int l = 0; l < MSA_LANE_GROUP; ++l) {
            field[l] = h;
        }
        for (int base = pstart; base < pend; base += MSA_CHUNK) {
            msa_field_t jj[MSA_CHUNK];
            u64 sw[MSA_CHUNK];
            #pragma unroll
            for (int q = 0; q < MSA_CHUNK; ++q) {
                int p = base + q;
                bool ok = p < pend;
                int nb = ok ? __ldg(&col_ind[p]) : 0;
                jj[q] = ok ? (msa_field_t)__ldg(&J[p]) : 0;
                sw[q] = st.at(nb, w);
            }
            #pragma unroll
            for (int l = 0; l < MSA_LANE_GROUP; ++l) {
                #pragma unroll
                for (int q = 0; q < MSA_CHUNK; ++q) {
                    field[l] += ((sw[q] >> (lane0 + l)) & 1ull) ? -jj[q] : jj[q];
                }
            }
        }
        #pragma unroll
        for (int l = 0; l < MSA_LANE_GROUP; ++l) {
#ifdef QUIP_MSA_COEF_F32
            // Shared draws couple real-valued replicas until all reads can
            // collapse into one local minimum. Give each replica its own draw.
            LaneTest lane_test = test;
            lane_test.draw = splitmix64(test.draw ^ (u64)(w * 64 + lane0 + l));
            lane_test.arm();
            accept |= (u64)lane_test.ok(field[l], bi, lane0 + l) << (lane0 + l);
#else
            accept |= (u64)test.ok(field[l], bi, lane0 + l) << (lane0 + l);
#endif
        }
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
                    msa_field_t h = __ldg(&mdl.h[var]);

#ifdef QUIP_MSA_WEIGHTED
                    if (pend - pstart > MSA_CHUNK) {
                        LaneTest test;
                        // Same draw as the single-pass path below.
                        test.draw = splitmix64(
                            mdl.slot_seed ^ ((u64)beta_idx << 40)
                            ^ ((u64)sweep << 20) ^ (u64)var);
#ifdef QUIP_MSA_LANE_I32
                        test.rung_beta = rung_beta;
                        test.rung_cut = rung_cut;
                        test.neglog_table = mdl.neglog_table;
#else
                        test.beta = beta;
#endif
                        u64 bi = st.at(var, w);
                        st.at(var, w) = bi ^ weighted_accept_dense<SPILL>(
                            mdl.csr_col_ind, mdl.J, st, bi, w, pstart, pend, h, test);
                        continue;
                    }
                    // Prefetch every neighbour index and coupling before
                    // touching spin state. weighted_accept_dense masks the
                    // same chunk load; the single-pass path stays inline
                    // (a shared helper measured +8.6% to +10.5% on
                    // weighted-i32 against main).
                    int nb[MSA_CHUNK];
                    msa_field_t jj[MSA_CHUNK];
                    #pragma unroll
                    for (int q = 0; q < MSA_CHUNK; ++q) {
                        int p = pstart + q;
                        bool ok = p < pend;
                        nb[q] = ok ? __ldg(&mdl.csr_col_ind[p]) : 0;
                        jj[q] = ok ? (msa_field_t)__ldg(&mdl.J[p]) : 0;
                    }

                    u64 bi = st.at(var, w);
                    u64 sw[MSA_CHUNK];
                    #pragma unroll
                    for (int q = 0; q < MSA_CHUNK; ++q) {
                        sw[q] = st.at(nb[q], w);
                    }
                    const u64 draw = splitmix64(
                        mdl.slot_seed ^ ((u64)beta_idx << 40)
                        ^ ((u64)sweep << 20) ^ (u64)var);
#ifdef QUIP_MSA_LANE_I32
                    const long long bound = __ldg(&mdl.neglog_table[draw & NEGLOG_MASK]);
#elif !defined(QUIP_MSA_COEF_F32)
                    // 23 bits keep k + 0.5f exact, so u never reaches 1.0f.
                    const float u = ((float)(draw >> 41) + 0.5f) * 1.1920929e-7f;
#endif
                    u64 accept = 0ull;
                    for (int lane = 0; lane < 64; ++lane) {
                        msa_field_t field = h;
                        #pragma unroll
                        for (int q = 0; q < MSA_CHUNK; ++q) {
                            // Spin bit 1 is -1. Missing neighbours have J = 0.
                            field += ((sw[q] >> lane) & 1ull) ? -jj[q] : jj[q];
                        }
#ifdef QUIP_MSA_COEF_F32
                        LaneTest test;
                        test.draw = splitmix64(draw ^ (u64)(w * 64 + lane));
                        test.beta = beta;
                        test.arm();
                        const bool ok = test.ok(field, bi, lane);
#else
                        const msa_field_t de = ((bi >> lane) & 1ull) ? 2 * field : -2 * field;
#ifdef QUIP_MSA_LANE_I32
                        const bool ok = de <= 0
                            || (de <= rung_cut && (long long)de * rung_beta < bound);
#else
                        const bool ok = de <= 0 || u < __expf(-beta * (float)de);
#endif
#endif
                        accept |= (u64)ok << lane;
                    }
                    st.at(var, w) = bi ^ accept;
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

// Dense Binary MSA: one 256-thread block per instance, three independent
// row-major i8 coefficient matrices (one per slot). Keeping all three copies
// preserves overlapped uploads and the sparse slot/control protocol. i8,
// rather than packed signs, keeps zero/diagonal entries explicit and leaves
// msa_coef_t as the coefficient seam for weighted lanes. This trades capacity
// for simple uploads and direct coefficient loads.
//
// Visit spins in fixed index order. All threads pack one coefficient row into
// shared sign/presence bit vectors. Replica lanes then compute local fields
// with population counts and decide flips. Zero entries and diagonal terms
// are masked out. No colouring or degree-limited bit counter is used.
// Spins retain the 64-replicas-per-u64 layout (one or two words). A second,
// transposed bit vector packs 32 nodes per u32 for each replica, with replica
// as the contiguous axis. This makes field reads coalesced and replaces
// scalar neighbour updates with population counts over 32 bonds at a time.
// Both spin layouts live in global memory. Dynamic shared memory holds two
// packed coefficient rows and four ballot words, far below Turing's budget.
// RNG state depends only on seed, slot and replica. Initial spin words use
// a separate counter hash so increasing N does not advance the update RNG.
// Weighted builds keep the same matrix/spin layouts and slot protocol. All
// threads load 256 coefficients at a time into shared memory, then replica
// lanes sum signed coefficients in i32 or f32. Fixed-size tiles bound shared
// memory independently of N. LaneTest preserves the sparse acceptance rules.
extern "C" __global__ void cuda_msa_dense_self_feeding(
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
    int max_packed_size,
    unsigned int base_seed,
    int words,
    u64* spin_state,
    unsigned int* replica_spins
#ifdef QUIP_MSA_LANE_I32
    , const long long* __restrict__ beta_fixed
    , const int* __restrict__ delta_cut
    , const long long* __restrict__ neglog_table
#endif
) {
    extern __shared__ unsigned int dense_scratch[];
    const int node_words = (N + 31) / 32;
#ifndef QUIP_MSA_WEIGHTED
    unsigned int* row_signs = dense_scratch;
    unsigned int* row_present = row_signs + node_words;
    unsigned int* ballots = row_present + node_words; // 4 entries
#else
    msa_coef_t* row = (msa_coef_t*)dense_scratch;
    unsigned int* ballots = (unsigned int*)(row + 256); // 4 entries
#endif
    __shared__ int s_active_slot;
    __shared__ int s_abort;
    const int tid = threadIdx.x;
    const int nonce = blockIdx.x;
    if (nonce >= num_nonces) return;
    const int ctrl_base = nonce * CTRL_STRIDE;
    const int replicas = words * 64;
    u64* state = spin_state + (size_t)nonce * N * words;
    unsigned int* by_replica = replica_spins + (size_t)nonce * node_words * replicas;
    int active_slot = -1;
    if (tid == 0) {
        while (true) {
            for (int s = 0; s < 3; ++s) {
                if (atomicCAS((int*)&nonce_ctrl[ctrl_base + s], SLOT_READY, SLOT_ACTIVE) == SLOT_READY) {
                    active_slot = s;
                    break;
                }
            }
            if (active_slot >= 0) {
                nonce_ctrl[ctrl_base + CTRL_ACTIVE_SLOT] = active_slot;
                __threadfence();
                break;
            }
            if (nonce_ctrl[ctrl_base + CTRL_EXIT_NOW]) break;
            __nanosleep(10000);
        }
        s_active_slot = active_slot;
    }
    __syncthreads();
    active_slot = s_active_slot;
    while (active_slot >= 0) {
        const int slot = nonce * 3 + active_slot;
        const msa_coef_t* J = slot_J_vals + (size_t)slot * N * N;
        const msa_coef_t* h = slot_h_vals + (size_t)slot * N;
        const u64 seed = splitmix64(((u64)base_seed << 8) ^ (u64)slot);
        u64 rng = splitmix64(seed ^ ((u64)(tid + 1) * 0x9E3779B97F4A7C15ull));
        if (rng == 0) rng = 0xdeadbeefcafef00dull;
        for (int i = tid; i < N * words; i += blockDim.x) {
            state[i] = splitmix64(seed ^ ((u64)(i / words) << 2) ^ (u64)(i % words));
        }
        __syncthreads();
        // Build the transposed view once. Each replica lane owns its word.
        for (int w = 0; w < node_words; ++w) {
            if (tid == 0) s_abort = nonce_ctrl[ctrl_base + CTRL_EXIT_NOW] == EXIT_ABORT;
            __syncthreads();
            if (s_abort) break;
            if (tid < replicas) {
                unsigned int bits = 0;
                for (int bit = 0; bit < 32; ++bit) {
                    const int node = w * 32 + bit;
                    if (node < N) {
                        bits |= (unsigned int)((state[node * words + tid / 64] >> (tid % 64)) & 1ull) << bit;
                    }
                }
                by_replica[(size_t)w * replicas + tid] = bits;
            }
        }
        __syncthreads();
        for (int beta_idx = 0; beta_idx < num_betas && !s_abort; ++beta_idx) {
#ifndef QUIP_MSA_LANE_I32
            const float beta = beta_schedule[beta_idx];
#endif
            for (int sweep = 0; sweep < sweeps_per_beta && !s_abort; ++sweep) {
                for (int node = 0; node < N; ++node) {
                    if (tid == 0) s_abort = nonce_ctrl[ctrl_base + CTRL_EXIT_NOW] == EXIT_ABORT;
                    __syncthreads();
                    if (s_abort) break;
#ifndef QUIP_MSA_WEIGHTED
                    // Every warp votes, including padding lanes in the last
                    // tile, so each ballot has its complete participation mask.
                    for (int base = 0; base < N; base += blockDim.x) {
                        const int j = base + tid;
                        const int coef = j < N && j != node ? (int)J[(size_t)node * N + j] : 0;
                        const unsigned int signs = __ballot_sync(0xffffffffu, coef < 0);
                        const unsigned int present = __ballot_sync(0xffffffffu, coef != 0);
                        if ((tid & 31) == 0 && j < N) {
                            row_signs[j / 32] = signs;
                            row_present[j / 32] = present;
                        }
                    }
                    __syncthreads();
                    if (tid < replicas) {
                        int field = h[node];
                        for (int w = 0; w < node_words; ++w) {
                            const unsigned int present = row_present[w];
                            const unsigned int disagree = (by_replica[(size_t)w * replicas + tid] ^ row_signs[w]) & present;
                            field += __popc(present) - 2 * __popc(disagree);
                        }
                        const int spin = (state[node * words + tid / 64] >> (tid % 64)) & 1ull ? -1 : 1;
                        const int delta = -2 * spin * field;
                        // Always advance once per spin, even on downhill moves.
                        const float u = (float)((xs64(rng) >> 40) + 1ull) * (1.0f / 16777216.0f);
                        const bool flip = tid < num_reads && (delta <= 0 || u < __expf(-beta * (float)delta));
#else
                    msa_field_t field = h[node];
                    for (int base = 0; base < N; base += 256) {
                        const int j = base + tid;
                        row[tid] = j < N && j != node ? J[(size_t)node * N + j] : 0;
                        __syncthreads();
                        if (tid < replicas) {
                            for (int offset = 0; offset < 256 && base + offset < N; offset += 32) {
                                const unsigned int spins = by_replica[(size_t)((base + offset) / 32) * replicas + tid];
                                #pragma unroll
                                for (int bit = 0; bit < 32; ++bit) {
                                    const msa_field_t coef = row[offset + bit];
                                    field += (spins >> bit) & 1u ? -coef : coef;
                                }
                            }
                        }
                        __syncthreads();
                    }
                    if (tid < replicas) {
                        LaneTest test;
                        test.draw = xs64(rng);
#ifdef QUIP_MSA_LANE_I32
                        test.rung_beta = beta_fixed[beta_idx];
                        test.rung_cut = delta_cut[beta_idx];
                        test.neglog_table = neglog_table;
#else
                        test.beta = beta;
#endif
                        test.arm();
                        const bool flip = tid < num_reads && test.ok(field, state[node * words + tid / 64], tid % 64);
#endif
                        if (flip) by_replica[(size_t)(node / 32) * replicas + tid] ^= 1u << (node % 32);
                        const unsigned int mask = __ballot_sync(0xffffffffu, flip);
                        if ((tid & 31) == 0) ballots[tid / 32] = mask;
                    }
                    __syncthreads();
                    if (tid < words) {
                        const u64 flip = (u64)ballots[tid * 2] | ((u64)ballots[tid * 2 + 1] << 32);
                        state[node * words + tid] ^= flip;
                    }
                    __syncthreads();
                }
            }
        }
        if (!s_abort) {
            for (int r = tid; r < num_reads; r += blockDim.x) {
                signed char* out = slot_samples + ((size_t)slot * num_reads + r) * max_packed_size;
                for (int b = 0; b < max_packed_size; ++b) {
                    unsigned int byte = 0;
                    for (int bit = 0; bit < 8; ++bit) {
                        const int node = b * 8 + bit;
                        if (node < N) byte |= ((state[node * words + r / 64] >> (r % 64)) & 1ull) << bit;
                    }
                    out[b] = (signed char)byte;
                }
                slot_energies[(size_t)slot * num_reads + r] = 0; // host scores exact energy
            }
        }
        __syncthreads();
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            s_active_slot = -1;
            if (!s_abort) {
                nonce_ctrl[ctrl_base + active_slot] = SLOT_COMPLETE;
                __threadfence();
                if (!nonce_ctrl[ctrl_base + CTRL_EXIT_NOW]) {
                    while (true) {
                        for (int s = 0; s < 3; ++s) {
                            if (atomicCAS((int*)&nonce_ctrl[ctrl_base + s], SLOT_READY, SLOT_ACTIVE) == SLOT_READY) {
                                s_active_slot = s;
                                break;
                            }
                        }
                        if (s_active_slot >= 0 || nonce_ctrl[ctrl_base + CTRL_EXIT_NOW]) break;
                        __nanosleep(10000);
                    }
                    if (s_active_slot >= 0) {
                        nonce_ctrl[ctrl_base + CTRL_ACTIVE_SLOT] = s_active_slot;
                        __threadfence();
                    }
                }
            }
        }
        __syncthreads();
        active_slot = s_active_slot;
    }
}
