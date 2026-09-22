// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

// ==============================================================================
// CUDA NONCE-PARALLEL PROBE SCREEN
// ==============================================================================
// Ranks fresh PoW nonces by a short anneal so a miner can spend its full solve
// only on the deepest instances (the probe screen measured in quip-miner-metal
// docs/perf/2026-09-18-probe-screen-at-scale.md).
//
// Each block anneals one group of 32 lanes. A group holds npg = 32 >> reads_log2
// nonces with R = 1 << reads_log2 reads each; lane L belongs to nonce L % npg.
// Unlike msa.cu, where the 64 lanes of a word are replicas of ONE instance,
// every nonce here has its own couplings, so the coupling sign is a per-lane
// mask. Nonce k's sign bits are stored as bit k of one byte per edge
// (npg <= 8), and the lane mask is that byte times rep = sum_r 1 << (r * npg).
//
// Couplings are drawn on the device exactly as
// quip_protocol::chacha8::draw_ising_milli draws them: a ChaCha8 stream keyed by
// the nonce, the first n_nodes words spent on h (always 0 on the supported
// topologies), then one word per edge, J negative when (word & 1) == neg_bit.
// The host checks this bit-for-bit against the protocol draw
// (`quip-screen verify`).
//
// The Metropolis step is msa.cu's (Isakov et al. 2015): with c the coupling
// sign bit and b the spin bit, l = c ^ b_i ^ b_j marks a satisfied bond; the
// d incident bonds are summed into bit planes and a lane flips when
// L <= (d + M) / 2, M a geometric draw shared by all lanes, read from a
// per-rung threshold row at a random per-sweep offset. Spins update one colour
// class at a time. There is no field term (h = 0 is a host precondition).
//
// Energy per lane is computed on the device: E = nnz - 2 * (satisfied bonds),
// in energy units. `quip-screen verify` checks it against energy_milli.
//
// Shared memory per block: 64 u64 thresholds + n u32 spin words + nnz bytes of
// coupling signs (padded to 16) + SCR_ROW threshold row. 68.5 KB at 4,577 nodes
// and 41,514 edges, so the screen needs >= 69 KB opt-in shared memory per block
// (Volta and Ampere and newer; not Turing).

#define SCR_MAX_DEG   20
#define SCR_ROW       8192
#define SCR_ROW_MASK  8191
#define SCR_MAX_FIELD 63
#define SCR_PLANES    5
#define SCR_MAX_COUNT 31
#ifndef SCR_THREADS
#define SCR_THREADS 256
#endif

typedef unsigned int u32;
typedef unsigned long long u64;

extern "C" {

__device__ __forceinline__ u64 scr_xs64(u64 &s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}
__device__ __forceinline__ u64 scr_splitmix64(u64 x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
__device__ __forceinline__ u32 scr_rotl(u32 v, int c) { return (v << c) | (v >> (32 - c)); }
#define SCR_QR(a, b, c, d) \
    a += b; d ^= a; d = scr_rotl(d, 16); \
    c += d; b ^= c; b = scr_rotl(b, 12); \
    a += b; d ^= a; d = scr_rotl(d, 8);  \
    c += d; b ^= c; b = scr_rotl(b, 7);

// One ChaCha8 keystream block (64-bit block counter, stream 0), the
// quip-protocol reference: 4 double rounds, output = rounds + input.
__device__ __forceinline__ void scr_chacha8(const u32* key, u64 counter, u32* out) {
    const u32 s0 = 0x61707865u, s1 = 0x3320646eu, s2 = 0x79622d32u, s3 = 0x6b206574u;
    const u32 s12 = (u32)(counter & 0xffffffffull), s13 = (u32)(counter >> 32);
    u32 x0 = s0, x1 = s1, x2 = s2, x3 = s3;
    u32 x4 = key[0], x5 = key[1], x6 = key[2], x7 = key[3];
    u32 x8 = key[4], x9 = key[5], x10 = key[6], x11 = key[7];
    u32 x12 = s12, x13 = s13, x14 = 0u, x15 = 0u;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        SCR_QR(x0, x4, x8, x12); SCR_QR(x1, x5, x9, x13); SCR_QR(x2, x6, x10, x14); SCR_QR(x3, x7, x11, x15);
        SCR_QR(x0, x5, x10, x15); SCR_QR(x1, x6, x11, x12); SCR_QR(x2, x7, x8, x13); SCR_QR(x3, x4, x9, x14);
    }
    out[0] = x0 + s0;  out[1] = x1 + s1;  out[2] = x2 + s2;  out[3] = x3 + s3;
    out[4] = x4 + key[0]; out[5] = x5 + key[1]; out[6] = x6 + key[2]; out[7] = x7 + key[3];
    out[8] = x8 + key[4]; out[9] = x9 + key[5]; out[10] = x10 + key[6]; out[11] = x11 + key[7];
    out[12] = x12 + s12; out[13] = x13 + s13; out[14] = x14; out[15] = x15;
}

__device__ __forceinline__ void scr_csa(u32 &h, u32 &l, u32 a, u32 b, u32 c) {
    u32 u = a ^ b;
    h = (a & b) | (u & c);
    l = u ^ c;
}
// Per-lane count of 20 one-bit inputs into 5 bit planes (Harley-Seal tree).
__device__ __forceinline__ void scr_popcount20(const u32* x, u32* planes) {
    u32 ones = 0, twos = 0, fours = 0, eights = 0;
    u32 tA, tB, fA, fB, eA, eB, sA, sB;
    scr_csa(tA, ones, ones, x[0], x[1]);   scr_csa(tB, ones, ones, x[2], x[3]);   scr_csa(fA, twos, twos, tA, tB);
    scr_csa(tA, ones, ones, x[4], x[5]);   scr_csa(tB, ones, ones, x[6], x[7]);   scr_csa(fB, twos, twos, tA, tB);
    scr_csa(eA, fours, fours, fA, fB);
    scr_csa(tA, ones, ones, x[8], x[9]);   scr_csa(tB, ones, ones, x[10], x[11]); scr_csa(fA, twos, twos, tA, tB);
    scr_csa(tA, ones, ones, x[12], x[13]); scr_csa(tB, ones, ones, x[14], x[15]); scr_csa(fB, twos, twos, tA, tB);
    scr_csa(eB, fours, fours, fA, fB);
    scr_csa(sA, eights, eights, eA, eB);
    scr_csa(tA, ones, ones, x[16], x[17]); scr_csa(tB, ones, ones, x[18], x[19]); scr_csa(fA, twos, twos, tA, tB);
    scr_csa(eA, fours, fours, fA, 0u);
    sB = eights & eA; eights ^= eA;
    planes[0] = ones; planes[1] = twos; planes[2] = fours; planes[3] = eights; planes[4] = sA | sB;
}
// Lanes whose 5-bit counter is <= limit (bit-serial compare, LSB first).
__device__ __forceinline__ u32 scr_le(const u32* planes, int limit) {
    int bound = limit + 1;
    if (bound > SCR_MAX_COUNT) return ~0u;
    u32 ge = ~0u;
    #pragma unroll
    for (int k = 0; k < SCR_PLANES; ++k) {
        u32 set = 0u - (u32)((bound >> k) & 1);
        u32 p = planes[k];
        ge = (p & ge) | ((p ^ ge) & ~set);
    }
    return ~ge;
}

// grid = groups, block = SCR_THREADS, dynamic shared =
// (SCR_MAX_FIELD + 1) * 8 + n_nodes * 4 + round_up(nnz, 16) + SCR_ROW bytes.
__global__ void __launch_bounds__(SCR_THREADS) quip_screen_probe(
    const u32* __restrict__ keys,          // [groups * npg * 8] nonce key words, little-endian
    int reads_log2,                        // 2..5: 4..32 reads per nonce
    int neg_bit,                           // J is negative when (word & 1) == neg_bit
    int n_nodes,
    int nnz,                               // edges
    const unsigned int* __restrict__ adj,  // [n_nodes * SCR_MAX_DEG] (nb << 16 | edge), 0xffffffff = none
    const int* __restrict__ color_starts,
    const int* __restrict__ color_counts,
    const int* __restrict__ color_nodes,
    int num_colors,
    const int* __restrict__ edge_u,        // [nnz]
    const int* __restrict__ edge_v,        // [nnz]
    const u64* __restrict__ cut_table,     // [num_betas][64]
    int num_betas,
    int sweeps_per_beta,
    unsigned int base_seed,
    int* __restrict__ out_energy,          // [groups * 32] energy units per lane
    u32* __restrict__ dbg_mask,            // group 0 lane masks [nnz], or null
    u32* __restrict__ dbg_state            // group 0 spin words [n_nodes], or null
) {
    extern __shared__ u64 scr_smem[];
    u64* cut = scr_smem;
    u32* state = (u32*)(cut + (SCR_MAX_FIELD + 1));
    unsigned char* j8 = (unsigned char*)(state + n_nodes);
    unsigned char* row = j8 + ((nnz + 15) & ~15);
    __shared__ int lane_count[32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;
    const int group = blockIdx.x;
    const int npg = 32 >> reads_log2;
    const int reads = 1 << reads_log2;
    u32 rep = 0;
    for (int r = 0; r < reads; ++r) rep |= 1u << (r * npg);
    const u32 nmask = (1u << npg) - 1u;

    // ---- 1. coupling signs. A warp draws `reads` consecutive ChaCha blocks at
    // once: lane L computes block b + L / npg for nonce L % npg, so byte q of
    // the ballot holds block b + q for all npg nonces.
    {
        u32 key[8];
        const u32* k = keys + ((size_t)group * npg + (lane % npg)) * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) key[i] = __ldg(&k[i]);
        const long long first_word = n_nodes;
        const long long b0 = first_word >> 4;
        const long long b1 = ((long long)n_nodes + nnz - 1) >> 4;
        u32 out[16];
        for (long long b = b0 + (long long)warp * reads; b <= b1; b += (long long)nwarps * reads) {
            scr_chacha8(key, (u64)(b + lane / npg), out);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                u32 mask = __ballot_sync(0xffffffffu, (int)(out[i] & 1u) == neg_bit);
                if (lane < reads) {
                    long long e = (b + lane) * 16 + i - first_word;
                    if (e >= 0 && e < nnz) j8[e] = (unsigned char)((mask >> (lane * npg)) & nmask);
                }
            }
        }
    }
    if (tid < 32) lane_count[tid] = 0;

    // ---- 2. anneal
    const u64 slot_seed = scr_splitmix64(((u64)base_seed << 20) ^ (u64)group);
    u64 rng = scr_splitmix64(slot_seed ^ ((u64)(tid + 1) * 0x9E3779B97F4A7C15ull));
    if (rng == 0) rng = 0xdeadbeefcafef00dull;
    for (int i = tid; i < n_nodes; i += blockDim.x) state[i] = (u32)scr_xs64(rng);
    __syncthreads();

    for (int beta_idx = 0; beta_idx < num_betas; ++beta_idx) {
        if (tid <= SCR_MAX_FIELD) cut[tid] = __ldg(&cut_table[(size_t)beta_idx * (SCR_MAX_FIELD + 1) + tid]);
        __syncthreads();
        for (int i = tid; i < SCR_ROW; i += blockDim.x) {
            u64 u = scr_xs64(rng);
            int lo = 0, hi = SCR_MAX_FIELD + 1;
            #pragma unroll
            for (int step = 0; step < 6; ++step) {
                int mid = (lo + hi) >> 1;
                if (u < cut[mid]) lo = mid; else hi = mid;
            }
            row[i] = (unsigned char)lo;
        }
        __syncthreads();
        for (int sweep = 0; sweep < sweeps_per_beta; ++sweep) {
            const int off = (int)(scr_splitmix64(slot_seed ^ ((u64)beta_idx << 20) ^ (u64)sweep) & SCR_ROW_MASK);
            for (int c = 0; c < num_colors; ++c) {
                const int start = __ldg(&color_starts[c]);
                const int cnt = __ldg(&color_counts[c]);
                for (int k = tid; k < cnt; k += blockDim.x) {
                    const int var = __ldg(&color_nodes[start + k]);
                    // The node's padded neighbour row: 20 entries in five
                    // 16-byte loads.
                    const uint4* row4 = (const uint4*)(adj + (size_t)var * SCR_MAX_DEG);
                    const u32 bi = state[var];
                    u32 x[SCR_MAX_DEG];
                    int d = 0;
                    #pragma unroll
                    for (int v = 0; v < SCR_MAX_DEG / 4; ++v) {
                        const uint4 e4 = __ldg(row4 + v);
                        const u32 ev[4] = {e4.x, e4.y, e4.z, e4.w};
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const bool ok = ev[q] != 0xffffffffu;
                            const u32 nb = ev[q] >> 16, e = ev[q] & 0xffffu;
                            x[v * 4 + q] = ok ? (((u32)j8[e] * rep) ^ bi ^ state[nb]) : 0u;
                            d += ok;
                        }
                    }
                    u32 planes[SCR_PLANES];
                    scr_popcount20(x, planes);
                    const int m = row[(var + off) & SCR_ROW_MASK];
                    const int limit = (d + m) >> 1;
                    const u32 accept = (limit >= d) ? ~0u : scr_le(planes, limit);
                    state[var] = bi ^ accept;
                }
                __syncthreads();
            }
        }
    }

    // ---- 3. energy per lane
    {
        int cnt[32];
        #pragma unroll
        for (int l = 0; l < 32; ++l) cnt[l] = 0;
        for (int e = tid; e < nnz; e += blockDim.x) {
            const u32 p = ((u32)j8[e] * rep) ^ state[__ldg(&edge_u[e])] ^ state[__ldg(&edge_v[e])];
            #pragma unroll
            for (int l = 0; l < 32; ++l) cnt[l] += (p >> l) & 1u;
        }
        #pragma unroll
        for (int l = 0; l < 32; ++l) {
            int v = cnt[l];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            if (lane == 0) atomicAdd(&lane_count[l], v);
        }
    }
    __syncthreads();
    if (tid < 32) out_energy[(size_t)group * 32 + tid] = nnz - 2 * lane_count[tid];
    if (group == 0) {
        if (dbg_mask) for (int e = tid; e < nnz; e += blockDim.x) dbg_mask[e] = (u32)j8[e] * rep;
        if (dbg_state) for (int i = tid; i < n_nodes; i += blockDim.x) dbg_state[i] = state[i];
    }
}

}  // extern "C"
