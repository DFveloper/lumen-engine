// Temporary T05A remediation plumbing: shared transform is unavailable.
#pragma once
#include <complex>
#include <stdexcept>
#include <vector>
namespace vllm::multimodal {
inline void AudioFft(std::vector<std::complex<double>>&) {
  throw std::runtime_error("T05A remediation: shared FFT not implemented");
}
}  // namespace vllm::multimodal
