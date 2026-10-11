#include "vllm/multimodal/gemma4_image_processor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vllm::multimodal {
// HF a08ace4bbd97e721c98751deec37d87b026acadc:
// get_aspect_ratio_preserving_size and Gemma4ImageProcessor._preprocess.
Gemma4ImageProcessor::Gemma4ImageProcessor(int64_t n) : max_soft_tokens_(n) {
  if (n != 70 && n != 140 && n != 280 && n != 560 && n != 1120)
    throw std::invalid_argument("Gemma4 unsupported soft-token budget");
}
std::array<int64_t, 2> Gemma4ImageProcessor::TargetSize(int64_t h, int64_t w) const {
  if (h <= 0 || w <= 0 || h > 65536 || w > 65536 || h * w > 64 * 1024 * 1024)
    throw std::invalid_argument("Gemma4 decoded image dimensions exceed bounded API");
  const double scale = std::sqrt(static_cast<double>(max_soft_tokens_ * 2304) /
                                 static_cast<double>(h * w));
  int64_t th = static_cast<int64_t>(std::floor(scale * h / 48)) * 48;
  int64_t tw = static_cast<int64_t>(std::floor(scale * w / 48)) * 48;
  if (th == 0) {
    th = 48;
    tw = std::min((w / h) * 48, max_soft_tokens_ * 48);
  } else if (tw == 0) {
    tw = 48;
    th = std::min((h / w) * 48, max_soft_tokens_ * 48);
  }
  if (th <= 0 || tw <= 0 || th * tw > max_soft_tokens_ * 2304)
    throw std::invalid_argument("Gemma4 invalid target geometry");
  return {th, tw};
}
Gemma4ImageFeatures Gemma4ImageProcessor::ProcessImage(std::span<const uint8_t> rgb,
                                                      int64_t h, int64_t w) const {
  const auto target = TargetSize(h, w);
  if (rgb.size() != static_cast<size_t>(h * w * 3))
    throw std::invalid_argument("Gemma4 decoded RGB byte count mismatch");
  if (target != std::array<int64_t, 2>{h, w})
    throw std::runtime_error("Gemma4 requires unsupported torchvision bicubic antialias resize");
  Gemma4ImageFeatures out;
  auto& patches = out.patches;
  patches.num_patches = max_soft_tokens_ * 9;
  patches.patch_feature_dim = 768;
  patches.image_grid_thw = {1, h / 16, w / 16};
  patches.pixel_values_f32.resize(static_cast<size_t>(patches.num_patches * 768), 0);
  out.position_ids.resize(static_cast<size_t>(patches.num_patches * 2), -1);
  out.num_soft_tokens = h * w / 2304;
  for (int64_t y = 0; y < h; ++y) {
    for (int64_t x = 0; x < w; ++x) {
      const int64_t row = (y / 16) * (w / 16) + x / 16;
      out.position_ids[row * 2] = x / 16;
      out.position_ids[row * 2 + 1] = y / 16;
      for (int64_t c = 0; c < 3; ++c)
        patches.pixel_values_f32[row * 768 + (y % 16) * 48 + (x % 16) * 3 + c] =
            static_cast<float>(rgb[(y * w + x) * 3 + c]) * (1.0f / 255.0f);
    }
  }
  return out;
}
}  // namespace vllm::multimodal
