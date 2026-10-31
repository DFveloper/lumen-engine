// Pascal tiled DP4A K-quant PREFILL (BACKEND-CUDA-SM061).
//
// The generic QuantDotGemmKernel maps one warp to one output (i, j). At prefill
// (m > 1) that re-reads weight row j once per prompt token and activation row i
// once per output column, so both sides stream from HBM m times / n times. This
// lane tiles the output as TILE_I columns x TILE_J rows: the weight tile and the
// activation tile are staged ONCE per 256-element K iteration in shared memory
// and reused across the tile, which is the shape llama.cpp's DP4A MMQ uses
// (ggml/src/ggml-cuda/mmq.cuh:386-403, mmq-load-tiles.cuh:711/:946).
//
// The config mirrors the pinned oracle's Pascal DP4A table:
//   ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh:100-110 (Q4_K) / :136-146
//   (Q6_K): nthreads=256, occupancy=2, I=64, J in {8..64}, MMQ_ITER_K=256.
// I and J are clamped to the oracle's J set for narrow shapes so a small N or M
// never asks for a tile larger than the matrix.
//
// NUMERIC CONTRACT. The activation stays the BlockQ8_K super-block the generic
// kernel and the CPU oracle both use (cpu_quant_act.cpp QuantizeRowQ8_K), and
// each staged weight block is consumed by the SAME DotSuperblock<W> the generic
// kernel calls. The integer DP4A core is therefore bit-identical to the generic
// lane and to the CPU reference; only the per-super-block float accumulation is
// reassociated (a per-thread sequential sum here instead of the generic lane's
// warp-tree reduce), exactly the near-tie band the landed tests already gate at
// NMSE 1e-6. Q6_K uses DotQ6KPrefill, a four-byte slide over the same ql/qh
// fields that feeds __dp4a directly instead of DotQ6K's 256-byte decode buffer.
// It groups the identical products as scale * sum(dp4a) rather than
// sum(scale * product); both are exact int32, so the integer core and the
// super-block float fold are unchanged. Deliberately NOT the oracle's Q8_1
// activation: that would change the integer core, and this row's acceptance
// requires the prefill token gate to stay exact against the landed Q8_K
// contract.
//
// Fallback: the caller keeps the generic lane for m == 1 (decode), for every
// dtype other than Q4_K/Q6_K, for non-Pascal devices, and when the shared-memory
// tile does not fit. Nothing here touches the Volta or newer GPU defaults.
#pragma once

namespace {

// Tile width in output columns (n). The prompt-row tile J is chosen at launch
// from the oracle's J set: 64 is the widest and the largest that fits Pascal's
// 48 KiB shared-memory budget at its Q4_K (144 B) / Q6_K (210 B) SRAM layout
// with a 64-row BlockQ8_K activation tile (64*210 + 64*292 = 32128 B).
constexpr int kPascalKQuantTileI = 64;
constexpr int kPascalKQuantThreads = 256;

// The oracle's Pascal DP4A table offers J in {8..64}. A narrow prompt takes the
// smaller J so the staged activation tile and its zero-fill shrink with it.
constexpr int PascalKQuantTileJFor(int64_t m) {
  return m <= 8 ? 8 : m <= 16 ? 16 : m <= 32 ? 32 : 64;
}

template <WType W>
struct PascalKQuantPrefillTraits;
template <>
struct PascalKQuantPrefillTraits<WType::kQ4_K> {
  using Block = BlockQ4_K;
  static constexpr int kBytes = sizeof(BlockQ4_K);
};
template <>
struct PascalKQuantPrefillTraits<WType::kQ6_K> {
  using Block = BlockQ6_K;
  static constexpr int kBytes = sizeof(BlockQ6_K);
};

template <WType W, int TileJ>
constexpr int PascalKQuantPrefillSmemBytes() {
  return kPascalKQuantTileI * PascalKQuantPrefillTraits<W>::kBytes +
         TileJ * static_cast<int>(sizeof(BlockQ8_K));
}

// Default on for Q4_K/Q6_K prefill on Pascal. VT_PASCAL_KQUANT_PREFILL=0
// restores the generic lane for an A/B rollback. Read per call so the
// in-process CUDA tests and a captured graph pick the choice up at launch.
inline bool PascalKQuantPrefillEnabled() {
  const char* e = std::getenv("VT_PASCAL_KQUANT_PREFILL");
  return !(e != nullptr && e[0] == '0' && e[1] == '\0');
}

// Eligibility is the caller's shape/device/dtype gate. sm_major == 6 is Pascal
// (sm_60/sm_61, the two devices whose DP4A the config table targets); sm_70
// (Volta) and every newer GPU keep the generic lane.
inline bool IsPascalKQuantPrefillEligible(const Tensor& out, const Tensor& a,
                                          const Tensor& b, int64_t nsb) {
  if (a.shape[0] <= 1 || nsb <= 0) return false;
  if (b.dtype != DType::kQ4_K && b.dtype != DType::kQ6_K) return false;
  if (!a.IsContiguous() || !b.IsContiguous()) return false;
  // Q4_K stages 16-byte chunks. The staged row stride is a whole number of
  // 144-byte blocks, so only the weight base needs the alignment guard; an
  // unaligned base falls back to the generic lane instead of a faulting load.
  if (b.dtype == DType::kQ4_K &&
      (reinterpret_cast<uintptr_t>(b.data) & 15) != 0) {
    return false;
  }
  if (out.dtype != DType::kF32 && out.dtype != DType::kBF16) return false;
  const auto& caps = GetDeviceCaps();
  if (!caps.valid || caps.sm_major != 6) return false;
  const int need = b.dtype == DType::kQ4_K
                       ? PascalKQuantPrefillSmemBytes<WType::kQ4_K, 64>()
                       : PascalKQuantPrefillSmemBytes<WType::kQ6_K, 64>();
  return need <= caps.max_shared_memory_per_block_optin;
}

// Four consecutive bytes out of a 2-byte-aligned Q6_K field. Q6_K blocks are
// 210 bytes, so a block base alternates 2-byte alignment and a plain int load is
// not legal; two aligned u16 reads build the same int (the GetIntB2 idiom).
__device__ __forceinline__ int PascalKQuantLoadI32(const uint8_t* p) {
  const uint16_t* x16 = reinterpret_cast<const uint16_t*>(p);
  return static_cast<int>(x16[0]) | (static_cast<int>(x16[1]) << 16);
}

// Q6_K DP4A dot for one 256-value super-block, matching DotQ6K's integer core.
// DotQ6K materializes a 256-byte int8 decode buffer per call; this walks the
// same ql/qh bit fields four bytes at a time and feeds __dp4a, then applies the
// per-16 scale. scale * sum(dp4a results) == sum(scale * product) in exact int32,
// so the integer core is bit-identical; only the caller's float fold differs, as
// it already does between the generic warp-tree and this tile-sequential sum.
__device__ inline float DotQ6KPrefill(const BlockQ6_K* xb, const BlockQ8_K* yb) {
  const int8_t* q8 = yb->qs;
  int isum = 0;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const uint8_t* ql = xb->ql + j * 64;
    const uint8_t* qh = xb->qh + j * 32;
    const uint8_t* q8c = reinterpret_cast<const uint8_t*>(q8 + j * 128);
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      const int scale = xb->scales[j * 8 + g];
      const int base = g * 16;
      int sub = 0;
#pragma unroll
      for (int t = 0; t < 16; t += 4) {
        const int L = base + t;
        int v;
        if (L < 32) {
          v = (PascalKQuantLoadI32(ql + L) & 0x0F0F0F0F) |
              ((PascalKQuantLoadI32(qh + L) & 0x03030303) << 4);
        } else if (L < 64) {
          v = (PascalKQuantLoadI32(ql + L) & 0x0F0F0F0F) |
              (((PascalKQuantLoadI32(qh + (L - 32)) >> 2) & 0x03030303) << 4);
        } else if (L < 96) {
          v = ((PascalKQuantLoadI32(ql + (L - 64)) >> 4) & 0x0F0F0F0F) |
              (((PascalKQuantLoadI32(qh + (L - 64)) >> 4) & 0x03030303) << 4);
        } else {
          v = ((PascalKQuantLoadI32(ql + (L - 64)) >> 4) & 0x0F0F0F0F) |
              (((PascalKQuantLoadI32(qh + (L - 96)) >> 6) & 0x03030303) << 4);
        }
        v = __vsub4(v, 0x20202020);
        sub = LegacyDp4a(v, PascalKQuantLoadI32(q8c + L), sub);
      }
      isum += scale * sub;
    }
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  return d * static_cast<float>(isum);
}

template <WType W>
__device__ inline float PascalKQuantPrefillDot(const void* w, const BlockQ8_K* a);
template <>
__device__ inline float PascalKQuantPrefillDot<WType::kQ4_K>(const void* w,
                                                             const BlockQ8_K* a) {
  return DotQ4K(static_cast<const BlockQ4_K*>(w), a);
}
template <>
__device__ inline float PascalKQuantPrefillDot<WType::kQ6_K>(const void* w,
                                                             const BlockQ8_K* a) {
  return DotQ6KPrefill(static_cast<const BlockQ6_K*>(w), a);
}

// One block computes a (TILE_I output columns) x (TileJ prompt rows) tile.
// TILE_I*TileJ outputs split evenly across kPascalKQuantThreads (16 each at
// 64x64). Each K super-block stages its weight tile and activation tile into
// shared memory, every thread accumulates its outputs with the same dot the
// generic lane uses, then the tile is overwritten on the next iteration.
template <WType W, typename OutT, int TileJ>
__global__ void PascalKQuantPrefillKernel(OutT* __restrict__ out,
                                          const uint8_t* __restrict__ weight,
                                          const BlockQ8_K* __restrict__ act,
                                          int64_t m, int64_t n, int64_t nsb,
                                          size_t w_row_bytes) {
  using Traits = PascalKQuantPrefillTraits<W>;
  using Block = typename Traits::Block;
  constexpr int kBB = Traits::kBytes;
  constexpr int kOutPerThread =
      (kPascalKQuantTileI * TileJ) / kPascalKQuantThreads;

  extern __shared__ __align__(16) uint8_t smem[];
  Block* sw = reinterpret_cast<Block*>(smem);
  BlockQ8_K* sa = reinterpret_cast<BlockQ8_K*>(smem + kPascalKQuantTileI * kBB);

  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * kPascalKQuantTileI;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * TileJ;
  const int tid = static_cast<int>(threadIdx.x);
  const int nt = static_cast<int>(blockDim.x);

  // TILE_I*TileJ outputs split across kPascalKQuantThreads. The mapping is
  // affine: tj is the thread's row within the tile and ti advances by
  // kPascalKQuantThreads/TileJ per output, so the K loop has no division.
  constexpr int kColStep = kPascalKQuantThreads / TileJ;
  const int tj = tid % TileJ;
  const int ti_base = tid / TileJ;
  float acc[kOutPerThread];
#pragma unroll
  for (int o = 0; o < kOutPerThread; ++o) acc[o] = 0.0f;

  const uint8_t* act_bytes = reinterpret_cast<const uint8_t*>(act);
  for (int64_t sb = 0; sb < nsb; ++sb) {
    // Stage TILE_I weight super-blocks (one per output column). All threads
    // share the copy. Q4_K blocks are 144 bytes and the eligible weight base is
    // 16-byte aligned, so a uint4 copy is aligned; Q6_K blocks are only 2-byte
    // aligned, so those copy u16 words.
    if constexpr (W == WType::kQ4_K) {
      constexpr int kChunks = kBB / 16;
      for (int c = tid; c < kPascalKQuantTileI * kChunks; c += nt) {
        const int i = c / kChunks;
        const int off = (c % kChunks) * 16;
        if (col0 + i < n) {
          *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(&sw[i]) + off) =
              *reinterpret_cast<const uint4*>(
                  weight + static_cast<size_t>(col0 + i) * w_row_bytes +
                  static_cast<size_t>(sb) * kBB + off);
        } else {
          memset(reinterpret_cast<uint8_t*>(&sw[i]) + off, 0, 16);
        }
      }
    } else {
      constexpr int kWords = kBB / 2;
      for (int c = tid; c < kPascalKQuantTileI * kWords; c += nt) {
        const int i = c / kWords;
        const int off = (c % kWords) * 2;
        if (col0 + i < n) {
          *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(&sw[i]) + off) =
              *reinterpret_cast<const uint16_t*>(
                  weight + static_cast<size_t>(col0 + i) * w_row_bytes +
                  static_cast<size_t>(sb) * kBB + off);
        } else {
          *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(&sw[i]) + off) = 0;
        }
      }
    }
    // Stage TILE_J activation super-blocks (one per prompt row). A zeroed tail
    // keeps the shared tile readable for the out-of-range outputs that are
    // dropped at write-back.
    constexpr int kActWords = sizeof(BlockQ8_K) / 4;
    for (int c = tid; c < TileJ * kActWords; c += nt) {
      const int j = c / kActWords;
      const int off = (c % kActWords) * 4;
      uint8_t* dst = reinterpret_cast<uint8_t*>(&sa[j]) + off;
      if (row0 + j < m) {
        *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(
            act_bytes + ((static_cast<size_t>(row0 + j) * nsb + sb) *
                         sizeof(BlockQ8_K)) + off);
      } else {
        *reinterpret_cast<uint32_t*>(dst) = 0;
      }
    }
    __syncthreads();

    const bool row_ok = row0 + tj < m;
    if (row_ok) {
#pragma unroll 1
      for (int o = 0; o < kOutPerThread; ++o) {
        const int ti = ti_base + o * kColStep;
        if (col0 + ti < n) acc[o] += PascalKQuantPrefillDot<W>(&sw[ti], &sa[tj]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int o = 0; o < kOutPerThread; ++o) {
    const int ti = ti_base + o * kColStep;
    const int64_t row = row0 + tj;
    const int64_t col = col0 + ti;
    if (row < m && col < n) {
      const float v = FinalFactor<W>() * acc[o];
      if constexpr (sizeof(OutT) == 4) {
        out[row * n + col] = v;
      } else {
        out[row * n + col] = DF32ToBF16(v);
      }
    }
  }
}

template <WType W, int TileJ>
void LaunchPascalKQuantPrefillJ(Tensor& out, const uint8_t* weight,
                                const BlockQ8_K* act, int64_t m, int64_t n,
                                int64_t nsb, size_t w_row_bytes,
                                cudaStream_t s) {
  constexpr int kSmem = PascalKQuantPrefillSmemBytes<W, TileJ>();
  const dim3 block(kPascalKQuantThreads);
  const dim3 grid(
      static_cast<unsigned>((n + kPascalKQuantTileI - 1) / kPascalKQuantTileI),
      static_cast<unsigned>((m + TileJ - 1) / TileJ));
  if (out.dtype == DType::kF32) {
    PascalKQuantPrefillKernel<W, float, TileJ><<<grid, block, kSmem, s>>>(
        static_cast<float*>(out.data), weight, act, m, n, nsb, w_row_bytes);
  } else {
    PascalKQuantPrefillKernel<W, uint16_t, TileJ><<<grid, block, kSmem, s>>>(
        static_cast<uint16_t*>(out.data), weight, act, m, n, nsb, w_row_bytes);
  }
}

template <WType W>
void LaunchPascalKQuantPrefill(Tensor& out, const uint8_t* weight,
                               const BlockQ8_K* act, int64_t m, int64_t n,
                               int64_t nsb, size_t w_row_bytes,
                               cudaStream_t s) {
  switch (PascalKQuantTileJFor(m)) {
    case 8:
      LaunchPascalKQuantPrefillJ<W, 8>(out, weight, act, m, n, nsb, w_row_bytes, s);
      break;
    case 16:
      LaunchPascalKQuantPrefillJ<W, 16>(out, weight, act, m, n, nsb, w_row_bytes, s);
      break;
    case 32:
      LaunchPascalKQuantPrefillJ<W, 32>(out, weight, act, m, n, nsb, w_row_bytes, s);
      break;
    default:
      LaunchPascalKQuantPrefillJ<W, 64>(out, weight, act, m, n, nsb, w_row_bytes, s);
      break;
  }
}

}  // namespace
