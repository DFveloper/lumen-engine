# T10A: Register Gemma4 mm chat seam (refusing stub)

Owning spec: `.agents/specs/gemma4-e4b-google-baseline.md` §T10A
State: ACCEPTED — commit `b2f449ba3`, review `3c85635e` PASS

## Gap

T09 flipped `mm_inputs_ready` to `true`, so the server now treats
Gemma4 as a multimodal model and calls
`MultiModalChatRegistry::MakeSeam(ctx)`. No registration exists for
`Gemma4ForConditionalGeneration`, so `RaiseForUnregistered` throws
and the catch block installs a generic refusing seam.

A registered factory with a Gemma4-specific refusal message is the
prerequisite for all serving-level multimodal work. It validates the
install context (tokenizer, config, prompt renderer, image codec)
and returns a seam that refuses multimodal requests with a clear
message naming the architecture and the pending modalities.

## Scope

Create `src/vllm/entrypoints/openai/mm_chat_gemma4.cpp` with:
- A `MakeGemma4ChatSeam` factory that validates the install context
  and returns a `MultiModalChatSeam` whose `chat_fn` refuses
  multimodal requests with a clear message
- `REGISTER_VLLM_MM_CHAT(gemma4, "Gemma4ForConditionalGeneration", ...)`
- Add the source to `tests/CMakeLists.txt`

## Out of scope

Image/audio/video processing in the chat path. Full serving
integration. Real model weights. Placeholder expansion.
