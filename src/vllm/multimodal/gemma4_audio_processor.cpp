// Temporary T05A remediation plumbing; never accepts waveform processing.
#include "vllm/multimodal/gemma4_audio_processor.h"
#include <stdexcept>
namespace vllm::multimodal {
AudioKwargs Gemma4AudioProcessor::ProcessWaveform(const float*, int64_t, int) const {
  throw std::runtime_error("T05A remediation: waveform frontend not implemented");
}
}  // namespace vllm::multimodal
