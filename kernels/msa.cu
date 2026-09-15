// Multi-spin coded simulated annealing for the self-feeding CUDA session
// Same slot/control plane and output layout as
// `cuda_sa_self_feeding` in sa.cu, so the host session code is shared.
//
// Algorithm: Isakov, Zintchenko, Ronnow, Troyer (2015), as implemented by
// quip-miner-cpu's `sa_msc.rs`: 64 replicas live in the bits of one 64-bit
// word per spin, `words` words per spin. Couplings must be +-1 and |h| <= 1
// (the host guarantees this; the CPU kernel has the same precondition). The
// Metropolis test is the integer comparison `sat <= (deg + M) / 2` where M is
// a geometric draw shared by every replica, taken from a per-rung table with a
// per-sweep random offset. Spins are updated colour class by colour class
// (greedy colouring from the host), so no two simultaneously updated spins
// interact; each thread owns one word of one spin at a time.
#ifndef QUIP_MAX_NODES
#define QUIP_MAX_NODES 5000
#endif
#define SLOT_EMPTY    0
#define SLOT_READY    1
#define SLOT_ACTIVE   2
#define SLOT_COMPLETE 3
#define CTRL_STRIDE       8
#define CTRL_ACTIVE_SLOT  3
#define CTRL_EXIT_NOW     6
#define MSA_PLANES    6
#define MSA_MAX_COUNT 63
#define MSA_MAX_FIELD 63
#define MSA_ROW       8192
#define MSA_ROW_MASK  8191
#ifndef MSA_DIAG
#define MSA_DIAG 0   // 1: no threshold rows (M=0), 2: no spin updates, 3: neither
#endif
#define MSA_MAX_DEG   20   // Zephyr (advantage2) degree bound; larger graphs fall back on the host

typedef unsigned long long u64;

extern "C" {

__device__ __forceinline__ u64 xs64(u64 &s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}
__device__ __forceinline__ u64 splitmix64(u64 x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
// Bit-sliced ripple add of one 1-bit-per-lane value into a 6-plane counter.
__device__ __forceinline__ void add_plane(u64* planes, u64 l) {
    u64 carry = l;
    #pragma unroll
    for (int p = 0; p < MSA_PLANES; ++p) {
        u64 next = planes[p] & carry;
        planes[p] ^= carry;
        carry = next;
    }
}
// Carry-save adder: (h, l) = a + b + c per lane.
__device__ __forceinline__ void csa(u64 &h, u64 &l, u64 a, u64 b, u64 c) {
    u64 u = a ^ b;
    h = (a & b) | (u & c);
    l = u ^ c;
}
// Per-lane popcount of 21 one-bit inputs (missing inputs are zero words) into
// planes[0..5] = ones, twos, fours, eights, sixteens, 0 (Harley-Seal tree,
// ~100 ops instead of 21 x 18 for a ripple add).
__device__ __forceinline__ void popcount21(const u64* x, u64* planes) {
    u64 ones = 0, twos = 0, fours = 0, eights = 0;
    u64 tA, tB, fA, fB, eA, eB, sA, sB;
    csa(tA, ones, ones, x[0], x[1]);  csa(tB, ones, ones, x[2], x[3]);  csa(fA, twos, twos, tA, tB);
    csa(tA, ones, ones, x[4], x[5]);  csa(tB, ones, ones, x[6], x[7]);  csa(fB, twos, twos, tA, tB);
    csa(eA, fours, fours, fA, fB);
    csa(tA, ones, ones, x[8], x[9]);  csa(tB, ones, ones, x[10], x[11]); csa(fA, twos, twos, tA, tB);
    csa(tA, ones, ones, x[12], x[13]); csa(tB, ones, ones, x[14], x[15]); csa(fB, twos, twos, tA, tB);
    csa(eB, fours, fours, fA, fB);
    csa(sA, eights, eights, eA, eB);
    csa(tA, ones, ones, x[16], x[17]); csa(tB, ones, ones, x[18], x[19]); csa(fA, twos, twos, tA, tB);
    tA = ones & x[20]; ones ^= x[20];
    fB = twos & tA;    twos ^= tA;
    csa(eA, fours, fours, fA, fB);
    sB = eights & eA;  eights ^= eA;
    planes[0] = ones; planes[1] = twos; planes[2] = fours; planes[3] = eights;
    planes[4] = sA | sB; planes[5] = 0ull;
}
// Lanes whose 6-bit counter is <= limit (bit-serial compare, LSB first).
__device__ __forceinline__ u64 le_constant(const u64* planes, int limit) {
    int bound = limit + 1;
    if (bound > MSA_MAX_COUNT) return ~0ull;
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

__global__ void cuda_msa_self_feeding(
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,
    const int* __restrict__ color_starts,
    const int* __restrict__ color_counts,
    const int* __restrict__ color_nodes,
    int num_colors,
    const signed char* __restrict__ slot_J_vals,
    const signed char* __restrict__ slot_h_vals,
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
    int words
) {
    extern __shared__ u64 smem[];
    u64* state = smem;                                   // N * words
    u64* cut = smem + (size_t)N * (size_t)words;         // MSA_MAX_FIELD + 1
    unsigned char* row = (unsigned char*)(cut + (MSA_MAX_FIELD + 1)); // MSA_ROW
    __shared__ int s_active_slot;
    __shared__ int s_abort;

    int tid = threadIdx.x;
    int nonce_id = blockIdx.x;
    if (nonce_id >= num_nonces) return;
    int ctrl_base = nonce_id * CTRL_STRIDE;

    int active_slot = -1;
    if (tid == 0) {
        while (true) {
            for (int s = 0; s < 3; s++) {
                int old = atomicCAS((int*)&nonce_ctrl[ctrl_base + s], SLOT_READY, SLOT_ACTIVE);
                if (old == SLOT_READY) { active_slot = s; break; }
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
    if (active_slot < 0) return;

    const int w = tid & (words - 1);          // words is 1, 2 or 4
    const int g = tid / words;
    const int gstride = blockDim.x / words;
    const int total = N * words;
    const int packed_size = (N + 7) / 8;

    while (true) {
        int slot_idx = nonce_id * 3 + active_slot;
        const signed char* my_J = &slot_J_vals[(long long)slot_idx * nnz];
        const signed char* my_h = &slot_h_vals[(long long)slot_idx * N];
        long long sample_base = (long long)slot_idx * num_reads * max_packed_size;
        long long energy_base = (long long)slot_idx * num_reads;
        u64 slot_seed = splitmix64(((u64)base_seed << 8) ^ (u64)(nonce_id * 3 + active_slot));
        u64 rng = splitmix64(slot_seed ^ ((u64)(tid + 1) * 0x9E3779B97F4A7C15ull));
        if (rng == 0) rng = 0xdeadbeefcafef00dull;

        for (int i = tid; i < total; i += blockDim.x) state[i] = xs64(rng);
        __syncthreads();

        bool aborted = false;
        for (int beta_idx = 0; beta_idx < num_betas; ++beta_idx) {
            // Abort-on-cancel: uniform decision via shared memory so every
            // thread leaves the loop at the same rung.
            if ((beta_idx & 7) == 0) {
                if (tid == 0) s_abort = nonce_ctrl[ctrl_base + CTRL_EXIT_NOW];
                __syncthreads();
                if (s_abort) { aborted = true; break; }
            }
            float beta = __ldg(&beta_schedule[beta_idx]);
            if (tid <= MSA_MAX_FIELD) {
                double p = exp(-2.0 * (double)beta * (double)tid);
                cut[tid] = (p >= 1.0) ? ~0ull : (u64)(p * 18446744073709551616.0);
            }
            __syncthreads();
#if MSA_DIAG == 1 || MSA_DIAG == 3
            for (int i = tid; i < MSA_ROW; i += blockDim.x) row[i] = 0;
#else
            for (int i = tid; i < MSA_ROW; i += blockDim.x) {
                u64 u = xs64(rng);
                int m = 0;
                if (u < cut[1]) {
                    m = 1;
                    while (m < MSA_MAX_FIELD && u < cut[m + 1]) ++m;
                }
                row[i] = (unsigned char)m;
            }
#endif
            __syncthreads();
            for (int sweep = 0; sweep < sweeps_per_beta; ++sweep) {
                int off = (int)(splitmix64(slot_seed ^ ((u64)beta_idx << 20) ^ (u64)sweep) & MSA_ROW_MASK);
#if MSA_DIAG == 2 || MSA_DIAG == 3
                for (int c = 0; c < num_colors; ++c) { __syncthreads(); }
                if (0)
#endif
                for (int c = 0; c < num_colors; ++c) {
                    int start = __ldg(&color_starts[c]);
                    int cnt = __ldg(&color_counts[c]);
                    for (int k = g; k < cnt; k += gstride) {
                        int var = __ldg(&color_nodes[start + k]);
                        int pstart = __ldg(&csr_row_ptr[var]);
                        int pend = __ldg(&csr_row_ptr[var + 1]);
                        int h = __ldg(&my_h[var]);
                        // Prefetch every neighbour index and coupling (independent
                        // global loads, L1-resident) before touching shared memory.
                        int nb[MSA_MAX_DEG];
                        int jj[MSA_MAX_DEG];
                        #pragma unroll
                        for (int q = 0; q < MSA_MAX_DEG; ++q) {
                            int p = pstart + q;
                            bool ok = p < pend;
                            nb[q] = ok ? __ldg(&csr_col_ind[p]) : 0;
                            jj[q] = ok ? (int)__ldg(&my_J[p]) : 0;
                        }
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

        if (!aborted) {
            // Pack replicas 0..num_reads-1 (bit 1 == spin -1, as sa.cu does).
            for (int r = tid; r < num_reads; r += blockDim.x) {
                int rw = r >> 6;
                if (rw >= words) rw = words - 1;
                int lane = r & 63;
                signed char* out = &slot_samples[sample_base + (long long)r * max_packed_size];
                for (int b = 0; b < packed_size; ++b) {
                    unsigned int byte = 0;
                    int base = b * 8;
                    #pragma unroll
                    for (int bit = 0; bit < 8; ++bit) {
                        int var = base + bit;
                        if (var < N) byte |= (unsigned int)((state[var * words + rw] >> lane) & 1ull) << bit;
                    }
                    out[b] = (signed char)byte;
                }
                slot_energies[energy_base + r] = 0;   // host rescores from spins
            }
        }
        __syncthreads();
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            if (nonce_ctrl[ctrl_base + CTRL_EXIT_NOW]) {
                s_active_slot = -1;           // never publish an aborted model
            } else {
                nonce_ctrl[ctrl_base + active_slot] = SLOT_COMPLETE;
                int next_slot = -1;
                while (true) {
                    for (int s = 0; s < 3; s++) {
                        int old = atomicCAS((int*)&nonce_ctrl[ctrl_base + s], SLOT_READY, SLOT_ACTIVE);
                        if (old == SLOT_READY) { next_slot = s; break; }
                    }
                    if (next_slot >= 0) break;
                    if (nonce_ctrl[ctrl_base + CTRL_EXIT_NOW]) break;
                    __nanosleep(10000);
                }
                if (next_slot >= 0) {
                    nonce_ctrl[ctrl_base + CTRL_ACTIVE_SLOT] = next_slot;
                    __threadfence();
                }
                s_active_slot = next_slot;
            }
        }
        __syncthreads();
        active_slot = s_active_slot;
        if (active_slot < 0) return;
    }
}
}  // extern "C"
