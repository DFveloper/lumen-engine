// Tests for the Gemma4 mm chat seam registration
// (mm_chat_gemma4.cpp), entered through
// `MultiModalChatRegistry::Find` and `MakeSeam`.
//
// T10A registers the seam. The seam refuses multimodal requests with
// a clear message naming the architecture. Text-only requests pass
// through unchanged.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <optional>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/multimodal/inputs.h"

namespace {

using namespace vllm::entrypoints::openai;

ChatCompletionRequest UserWith(std::vector<ChatCompletionContentPart> parts) {
  ChatCompletionRequest r;
  ChatCompletionMessage msg;
  msg.role = "user";
  msg.content = std::move(parts);
  r.messages.push_back(std::move(msg));
  return r;
}

ChatCompletionContentPart TextPart(std::string text) {
  ChatCompletionContentPart p;
  p.type = "text";
  p.text = std::move(text);
  return p;
}

ChatCompletionContentPart ImagePart(int w, int h) {
  ChatCompletionContentPart p;
  p.type = "image_url";
  ChatCompletionImageURL u;
  // Minimal data URL — the seam refuses before decoding.
  u.url = "data:image/png;base64,iVBOR";
  p.image_url = std::move(u);
  return p;
}

}  // namespace

// T10A RED: the seam is registered for the architecture.
TEST_CASE("gemma4 mm chat: seam is registered for Gemma4ForConditionalGeneration") {
  const MultiModalChatRegistration* reg =
      MultiModalChatRegistry::Find("Gemma4ForConditionalGeneration");
  REQUIRE(reg != nullptr);
  CHECK(reg->make_seam != nullptr);
}

// T10A: the seam refuses multimodal requests with a clear message.
TEST_CASE("gemma4 mm chat: multimodal request is refused with architecture name") {
  // Build a minimal context.
  MultiModalChatContext ctx;
  ctx.architecture = "Gemma4ForConditionalGeneration";

  const MultiModalChatSeam seam = MultiModalChatRegistry::MakeSeam(ctx);
  REQUIRE(seam.chat_fn);

  // A text-only request must pass through unchanged.
  auto text_out = seam.chat_fn({UserWith({TextPart("hello")})});
  CHECK_FALSE(text_out.has_value());

  // A request with an image must be refused.
  auto mm_out = seam.chat_fn({UserWith({ImagePart(48, 48), TextPart("what is this?")})});
  REQUIRE_FALSE(mm_out.has_value());
  CHECK_THAT(mm_out.error(),
             Catch::Matchers::Contains("Gemma4", Catch::CaseSensitive::No));
}
