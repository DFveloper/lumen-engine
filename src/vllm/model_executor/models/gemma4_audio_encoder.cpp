// Pinned Gemma4AudioModel.forward and gemma4_mm.py::_process_audio_input.
// CPU reference adapter only; embed_mm and production BF16 remain owed.
#include "vllm/model_executor/models/gemma4_loaded.h"
#include <cmath>
#include <cstring>

namespace vllm {
constinit const ModelFactory kGemma4MmHooks{.encode_mm = &EncodeMmGemma4Audio};
MmEncoderOutput EncodeMmGemma4Audio(LoadedModel& model, const HfConfig& config,
    vt::Queue& queue, const multimodal::MultiModalFeatureSpec& item) {
  VT_CHECK(queue.device.type == vt::DeviceType::kCPU, "gemma4 audio: CPU reference only");
  VT_CHECK(item.modality == "audio" && item.audio_data, "gemma4 encoder: audio only");
  const auto& w = ModelAs<Gemma4LoadedModel>(model, "Gemma4ForConditionalGeneration").weights();
  VT_CHECK(w.audio_config && w.audio_frontend_config, "gemma4 audio: tower absent");
  auto cfg = *w.audio_config;
  const auto& a = *item.audio_data;
  VT_CHECK(a.num_chunks == 1 && a.chunk_num_samples.empty() && a.chunk_num_tokens.empty(),
           "gemma4 audio: one unchunked clip required");
  VT_CHECK(a.n_mels > 0 && a.n_frames > 0 && a.n_frames <= INT64_MAX / a.n_mels &&
           static_cast<uint64_t>(a.n_frames) <= SIZE_MAX / sizeof(float) / a.n_mels &&
           a.input_features.size() == static_cast<size_t>(a.n_frames * a.n_mels),
           "gemma4 audio: feature shape");
  VT_CHECK(cfg.hidden_size % 2 == 0 && cfg.hidden_size % cfg.num_heads == 0 &&
           cfg.hidden_act == "silu" && cfg.context_right == 0 && cfg.context_left > 1,
           "gemma4 audio: unsupported forward config");
  VT_CHECK(a.n_mels <= INT64_MAX-3 && (a.n_mels+3)/4 == cfg.sub_ch0/4,
           "gemma4 audio: mel projection width");
  VT_CHECK(config.hidden_size == cfg.text_hidden_size, "gemma4 audio: text width");
  VT_CHECK(a.feature_mask.empty() || a.feature_mask.size() == static_cast<size_t>(a.n_frames),
           "gemma4 audio: mask length");
  int64_t valid = 0;
  bool padding = false;
  for (int64_t t=0; t<a.n_frames; ++t) {
    const int bit = a.feature_mask.empty() ? 1 : a.feature_mask[t];
    VT_CHECK(bit == 0 || bit == 1, "gemma4 audio: mask must be boolean");
    VT_CHECK(!padding || bit == 0, "gemma4 audio: mask must be valid prefix");
    if (bit) ++valid; else padding = true;
  }
  VT_CHECK(valid > 0, "gemma4 audio: no valid frames");
  const int64_t rows = (valid-1)/4+1;
  VT_CHECK(item.length == rows && (a.num_tokens == 0 || a.num_tokens == rows),
           "gemma4 audio: placeholder length");
  VT_CHECK(rows <= INT64_MAX / cfg.text_hidden_size &&
           static_cast<uint64_t>(rows) <= SIZE_MAX / sizeof(float) / cfg.text_hidden_size,
           "gemma4 audio: output size overflow");
  // Causal attention and masked convolutions: trim the invalid suffix before
  // encoding. This emits exactly mask[::2][::2]'s true rows, never padded rows.
  std::vector<float> features(static_cast<size_t>(valid)*a.n_mels);
  for (int64_t t=0; t<valid; ++t)
    for (int64_t m=0; m<a.n_mels; ++m) {
      const float value=a.input_features[m*a.n_frames+t];
      VT_CHECK(std::isfinite(value), "gemma4 audio: nonfinite features");
      features[t*a.n_mels+m]=value;
    }
  cfg.feature_size=a.n_mels;
  const auto& p=w.audio.embedding_projection;
  VT_CHECK(p.dtype == vt::DType::kBF16 && p.rank == 2 &&
           p.shape[0] == cfg.text_hidden_size && p.shape[1] == cfg.output_proj_dims &&
           p.bytes.size() == static_cast<size_t>(p.shape[0]*p.shape[1])*2,
           "gemma4 audio: BF16 connector shape");
  std::vector<float> connector(p.bytes.size()/2);
  for (size_t i=0; i<connector.size(); ++i) {
    uint16_t bits; std::memcpy(&bits,p.bytes.data()+2*i,2);
    connector[i]=vt::BF16ToF32(bits);
  }
  auto values=std::make_shared<std::vector<float>>(multimodal::Gemma4AudioForward(
      features, valid, {}, w.audio_frontend_weights, cfg, nullptr, &connector));
  VT_CHECK(values->size() == static_cast<size_t>(rows*cfg.text_hidden_size),
           "gemma4 audio: encoder output shape");
  return {values, vt::Tensor::Contiguous(values->data(), vt::DType::kF32,
                                        queue.device, {rows,cfg.text_hidden_size})};
}
}  // namespace vllm
