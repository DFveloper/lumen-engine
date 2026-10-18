// The `Gemma4ForConditionalGeneration` multimodal chat seam (row
// MODEL-MM-gemma4-mm-gemma4-for-conditional-generation,
// .agents/specs/gemma4-e4b-google-baseline.md).
//
// T10A registers the seam so the server recognises the architecture.
// The seam refuses multimodal requests with a clear message naming
// the architecture and the pending modalities. Text-only requests
// pass through unchanged (return nullopt).
//
// Image, audio, and video processing in the chat path is out of
// scope for T10A — it is the prerequisite registration for all
// serving-level multimodal work (T10B+).
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/multimodal/inputs.h"

namespace vllm::entrypoints::openai {
namespace {

MultiModalChatSeam MakeGemma4ChatSeam(const MultiModalChatContext& ctx) {
  // Validate the install context — the fields the image/audio/video
  // paths will need when they are implemented.
  if (ctx.tokenizer == nullptr || ctx.config == nullptr) {
    throw std::runtime_error(
        "Gemma4 multimodal chat seam: the install context is incomplete "
        "(tokenizer and resolved model config are required)");
  }

  MultiModalChatSeam seam;
  seam.detail =
      "Gemma4 multimodal chat seam (registered, image/audio/video pending)";
  seam.chat_fn = MakeRefusingMultiModalChatFn(
      "Gemma4ForConditionalGeneration",
      "Gemma4 multimodal chat: image, audio, and video processing in the "
      "chat path is not yet implemented. The model loads and the runner "
      "dispatch path is open (mm_inputs_ready=true), but the serving seam "
      "does not yet construct MultiModalFeatureSpec from chat messages.");
  return seam;
}

}  // namespace

REGISTER_VLLM_MM_CHAT(gemma4, "Gemma4ForConditionalGeneration",
                       &MakeGemma4ChatSeam)

}  // namespace vllm::entrypoints::openai
