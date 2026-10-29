// Pascal Q6_K MMQ numerical contract, llama-cpp 62318cd57:
// mmq.cuh:85-92; vecdotq.cuh:650-675; mmq-load-tiles.cuh:946-1030.
// Original compressed weights remain in canonical NK order.
template <typename OutT>
__global__ void PascalQ6Prefill(OutT* out, const uint8_t* weight,
                                const PascalMmqD4* act, int64_t m,
                                int64_t n, int64_t nsb) {
  const int64_t output = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (output >= m * n) return;
  const int64_t row = output / n, col = output % n;
  const int lane = threadIdx.x;
  float partial = 0;
  // Eight lanes consume the eight independently scaled activation groups.
  for (int64_t group = lane; group < nsb * 8; group += 32) {
    const int64_t b = group / 8;
    const int g = group % 8;
    const auto* w = reinterpret_cast<const BlockQ6_K*>(weight + (col * nsb + b) * 210);
    const auto& a = act[row * nsb * 8 + group];
    const int half = g / 4, quarter = g % 4;
    int first = 0, second = 0;
#pragma unroll
    for (int pack = 0; pack < 8; ++pack) {
      const int index = half * 64 + (quarter % 2) * 32 + pack * 4;
      const int vl = GetIntB2(reinterpret_cast<const int8_t*>(w->ql), index / 4);
      const int vh = GetIntB2(reinterpret_cast<const int8_t*>(w->qh), (half * 32 + pack * 4) / 4);
      const int lo = (vl >> (quarter >= 2 ? 4 : 0)) & 0x0f0f0f0f;
      const int hi = ((vh >> (quarter * 2)) & 0x03030303) << 4;
      const int value = __vsub4(lo | hi, 0x20202020);
      const int input = GetIntB2(a.qs, pack);
      if (pack < 4) first = LegacyDp4a(value, input, first);
      else second = LegacyDp4a(value, input, second);
    }
    const int integer = first * w->scales[g * 2] + second * w->scales[g * 2 + 1];
    partial += DF16ToF32(w->d) * (a.d * integer);
  }
#pragma unroll
  for (int mask = 16; mask; mask >>= 1)
    partial += __shfl_down_sync(0xffffffffu, partial, mask);
  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) out[output] = partial;
    else out[output] = DF32ToBF16(partial);
  }
}

void LaunchPascalQ6Prefill(Tensor& out, const Tensor& a, const Tensor& weight,
                            cudaStream_t stream) {
  const int64_t m = a.shape[0], n = weight.shape[0], nsb = a.shape[1] / 256;
  const auto* act = EnsurePascalMmqD4(a, m, nsb * 8, stream);
  const dim3 block(32, 4);
  const unsigned grid = static_cast<unsigned>((m * n + 3) / 4);
  if (out.dtype == DType::kF32)
    PascalQ6Prefill<<<grid, block, 0, stream>>>(static_cast<float*>(out.data),
        static_cast<const uint8_t*>(weight.data), act, m, n, nsb);
  else
    PascalQ6Prefill<<<grid, block, 0, stream>>>(static_cast<uint16_t*>(out.data),
        static_cast<const uint8_t*>(weight.data), act, m, n, nsb);
}
