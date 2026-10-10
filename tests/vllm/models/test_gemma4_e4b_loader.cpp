// Synthetic loader contract, not original-checkpoint inference. Original names
// follow the pinned Google config in fixtures/gemma4_e4b_google. Dimensions are
// reduced to avoid allocating the original vocabulary and PLE tables.
#include <doctest/doctest.h>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <random>
#include <nlohmann/json.hpp>
#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"

namespace {
using nlohmann::json;
struct Fixture {
  std::filesystem::path path;
  std::filesystem::path directory;
  json header = json::object();
  std::string bytes;
  Fixture() {
    for (int attempt = 0; attempt < 32; ++attempt) {
      directory = std::filesystem::temp_directory_path() /
          ("e4b-loader-" + std::to_string(std::random_device{}()));
      if (std::filesystem::create_directory(directory)) break;
      directory.clear();
    }
    if (directory.empty()) throw std::runtime_error("cannot create fixture directory");
    path = directory / "model.safetensors";
    add("model.language_model.embed_tokens.weight", {3, 2});
    add("model.language_model.norm.weight", {2});
    const std::string p = "model.language_model.layers.0.";
    for (const auto* n : {"input_layernorm", "post_attention_layernorm",
                         "pre_feedforward_layernorm", "post_feedforward_layernorm"})
      add(p + n + ".weight", {2});
    for (const auto* n : {"q_proj", "k_proj", "v_proj", "o_proj"})
      add(p + "self_attn." + n + ".weight", {2, 2});
    for (const auto* n : {"q_norm", "k_norm"})
      add(p + "self_attn." + n + ".weight", {2});
    for (const auto* n : {"gate_proj", "up_proj", "down_proj"})
      add(p + "mlp." + n + ".weight", {2, 2});
  }
  ~Fixture() { std::error_code ec; std::filesystem::remove_all(directory, ec); }
  void add(const std::string& name, std::vector<int64_t> shape) {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    const size_t start = bytes.size();
    // BF16 1.0, serialized little-endian.
    for (size_t i = 0; i < n; ++i) { bytes.push_back('\x80'); bytes.push_back('\x3f'); }
    header[name] = {{"dtype", "BF16"}, {"shape", shape},
                    {"data_offsets", {start, bytes.size()}}};
  }
  std::vector<vllm::SafetensorsFile> open() {
    auto h = header.dump();
    while (h.size() % 8) h.push_back(' ');
    std::ofstream f(path, std::ios::binary);
    for (unsigned i = 0; i < 8; ++i) f.put(static_cast<char>((uint64_t(h.size()) >> (8*i)) & 255));
    f << h << bytes;
    f.close();
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(path.string()));
    return shards;
  }
};
vllm::HfConfig config() {
  vllm::HfConfig c;
  c.hidden_size = 2; c.vocab_size = 3; c.num_hidden_layers = 1;
  c.num_attention_heads = 1; c.num_key_value_heads = 1; c.head_dim = 2;
  c.raw = {{"text_config", json::object()}};
  return c;
}
}
TEST_CASE("E4B loader: absent tie flag defaults to tied without head allocation") {
  Fixture f; auto shards = f.open();
  auto w = vllm::LoadGemma4ForConditionalGenerationWeights(shards, config());
  CHECK(w.tie_word_embeddings); CHECK(w.lm_head.bytes.empty());
  REQUIRE(w.layers.size() == 1); CHECK(w.layers[0].head_dim == 2);
  CHECK_FALSE(w.layers[0].is_kv_shared); CHECK_FALSE(w.layers[0].k_eq_v);
}
TEST_CASE("E4B loader: false tie flag requires lm_head") {
  Fixture f; auto shards = f.open(); auto c = config();
  c.raw["text_config"]["tie_word_embeddings"] = false;
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                    doctest::Contains("lm_head.weight"));
}
TEST_CASE("E4B loader: untied head has vocab by hidden source shape") {
  Fixture f; f.add("lm_head.weight", {3, 2});
  const uint16_t values[] = {0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0};
  const size_t start = f.bytes.size() - sizeof(values);
  for (size_t i = 0; i < 6; ++i) {
    f.bytes[start + 2*i] = static_cast<char>(values[i] & 255);
    f.bytes[start + 2*i + 1] = static_cast<char>(values[i] >> 8);
  }
  auto shards = f.open(); auto c = config();
  c.raw["text_config"]["tie_word_embeddings"] = false;
  auto w = vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
  CHECK_FALSE(w.tie_word_embeddings); CHECK(w.lm_head.rank == 2);
  CHECK(w.lm_head.dtype == vt::DType::kBF16);
  CHECK(w.lm_head.shape[0] == 2); CHECK(w.lm_head.shape[1] == 3);
  REQUIRE(w.lm_head.bytes.size() == 12);
  uint16_t actual[6]; std::memcpy(actual, w.lm_head.bytes.data(), sizeof(actual));
  const uint16_t expected[] = {0x3f80, 0x4040, 0x40a0, 0x4000, 0x4080, 0x40c0};
  for (size_t i = 0; i < 6; ++i) CHECK(actual[i] == expected[i]);
  std::vector<vllm::SafetensorsFile> second;
  second.push_back(vllm::SafetensorsFile::Open(f.path.string()));
  auto owned = std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(second));
  auto ow = vllm::LoadGemma4ForConditionalGenerationWeightsOwned(owned, c);
  CHECK(ow.lm_head.dtype == vt::DType::kBF16);
  REQUIRE(ow.lm_head.bytes.size() == w.lm_head.bytes.size());
  CHECK(std::memcmp(ow.lm_head.bytes.data(), w.lm_head.bytes.data(), 12) == 0);
}
TEST_CASE("E4B loader: correctly shaped F16 untied head is refused by both overloads") {
  Fixture f; f.add("lm_head.weight", {3, 2});
  // F16 has the same byte width: only the dtype changes, not shape or payload.
  f.header["lm_head.weight"]["dtype"] = "F16";
  auto shards = f.open(); auto c = config();
  c.raw["text_config"]["tie_word_embeddings"] = false;
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                    doctest::Contains("BF16"));
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                    doctest::Contains("lm_head.weight"));
  auto owned = std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards));
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeightsOwned(owned, c),
                    doctest::Contains("BF16"));
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeightsOwned(owned, c),
                    doctest::Contains("lm_head.weight"));
}
TEST_CASE("E4B loader: malformed untied head shape is refused") {
  for (const auto& shape : std::vector<std::vector<int64_t>>{{2, 3}, {6}, {3, 1}}) {
    Fixture f; f.add("lm_head.weight", shape); auto shards = f.open(); auto c = config();
    c.raw["text_config"]["tie_word_embeddings"] = false;
    CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                      doctest::Contains("lm_head.weight"));
  }
}
TEST_CASE("E4B loader: tied flag ignores an unused separate head") {
  Fixture f; f.add("lm_head.weight", {1}); auto shards = f.open(); auto c = config();
  c.raw["text_config"]["tie_word_embeddings"] = true;
  auto w = vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
  CHECK(w.tie_word_embeddings); CHECK(w.lm_head.bytes.empty());
}
TEST_CASE("E4B loader: invalid tie flag is not silently defaulted") {
  for (const auto& flag : std::vector<json>{"false", 0, json::array(), json::object()}) {
    Fixture f; auto shards = f.open(); auto c = config();
    c.raw["text_config"]["tie_word_embeddings"] = flag;
    CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                      doctest::Contains("tie_word_embeddings"));
  }
}
TEST_CASE("E4B loader: null tie flag retains the legacy default") {
  Fixture f; auto shards = f.open(); auto c = config();
  c.raw["text_config"]["tie_word_embeddings"] = nullptr;
  auto w = vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
  CHECK(w.tie_word_embeddings); CHECK(w.lm_head.bytes.empty());
}
TEST_CASE("E4B loader: head rejection uses original E4B config dimensions") {
  // No original-size tensor allocation: an intentionally tiny head is refused
  // against the independently pinned original config dimensions.
  const auto c0 = vllm::LoadHfConfig(E4B_CONFIG_PATH);
  CHECK(c0.hidden_size == 2560); CHECK(c0.vocab_size == 262144);
  CHECK(c0.num_hidden_layers == 42);
  CHECK(c0.raw["text_config"]["hidden_size_per_layer_input"] == 256);
  CHECK(c0.raw["text_config"]["num_kv_shared_layers"] == 18);
  CHECK(c0.raw["text_config"]["tie_word_embeddings"] == true);
  Fixture f; f.add("lm_head.weight", {3, 2}); auto shards = f.open(); auto c = c0;
  c.raw["text_config"]["tie_word_embeddings"] = false;
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                    doctest::Contains("lm_head.weight"));
}

namespace {
const std::string vision_proj = "model.embed_vision.embedding_projection.weight";
const std::string audio_proj = "model.embed_audio.embedding_projection.weight";
vllm::HfConfig mm_config() {
  auto c = config();
  c.raw["vision_config"] = {{"hidden_size", 3}, {"rms_norm_eps", 1e-6},
                            {"use_clipped_linears", true}};
  c.raw["audio_config"] = {{"hidden_size", 4}, {"output_proj_dims", 5},
                           {"rms_norm_eps", 1e-6}};
  return c;
}
void reject_mm(Fixture& f, const vllm::HfConfig& c, const std::string& diagnostic) {
  auto shards = f.open();
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                    doctest::Contains(diagnostic));
  auto owned = std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards));
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeightsOwned(owned, c),
                    doctest::Contains(diagnostic));
}
}
TEST_CASE("E4B projectors: configured modalities require their weights") {
  Fixture f; auto c = mm_config();
  reject_mm(f, c, vision_proj);
  f.add(vision_proj, {2, 3});
  reject_mm(f, c, audio_proj);
}
TEST_CASE("E4B projectors: malformed config is refused by both loaders") {
  for (const auto* key : {"vision_config", "audio_config"}) {
    for (const json& bad : std::vector<json>{false, 7, "bad", json::array()}) {
      Fixture f; auto c = config(); c.raw[key] = bad;
      reject_mm(f, c, key);
    }
    for (const auto* field : {"hidden_size", "rms_norm_eps"}) {
      for (const json& bad : std::vector<json>{nullptr, false, -1, 0, "bad"}) {
        Fixture f; auto c = config();
        c.raw[key] = {{"hidden_size", 3}, {"rms_norm_eps", 1e-6}};
        c.raw[key][field] = bad;
        reject_mm(f, c, field);
      }
    }
  }
  for (const json& bad : std::vector<json>{false, -1, 1.5, "bad"}) {
    Fixture f; auto c = config(); c.raw["audio_config"] = mm_config().raw["audio_config"];
    c.raw["audio_config"]["output_proj_dims"] = bad;
    reject_mm(f, c, "output_proj_dims");
  }
}
TEST_CASE("E4B projectors: exact rank and orientation are required") {
  for (const auto* key : {"vision_config", "audio_config"}) {
    const auto name = std::string(key) == "vision_config" ? vision_proj : audio_proj;
    for (const auto& shape : std::vector<std::vector<int64_t>>{{3, 2}, {6}, {2, 1}}) {
      Fixture f; auto c = config(); c.raw[key] = {{"hidden_size", 3}, {"rms_norm_eps", 1e-6}};
      f.add(name, shape); reject_mm(f, c, name);
    }
    Fixture f; auto c = config(); c.raw[key] = {{"hidden_size", 3}, {"rms_norm_eps", 1e-6}};
    f.add(name, {2, 3}); f.header[name]["dtype"] = "F16";
    reject_mm(f, c, "BF16");
  }
}
TEST_CASE("E4B projectors: owned raw BF16 layout and full config survive shard destruction") {
  for (bool owned : {false, true}) {
    vllm::Gemma4Weights w;
    auto c = mm_config();
    {
      Fixture f; f.add(vision_proj, {2, 3}); f.add(audio_proj, {2, 5});
      const size_t start = f.header[vision_proj]["data_offsets"][0].get<size_t>();
      for (size_t i = 0; i < 6; ++i) {
        f.bytes[start + 2*i] = 0; f.bytes[start + 2*i + 1] = char(0x3f + i);
      }
      auto shards = f.open();
      w = owned ? vllm::LoadGemma4ForConditionalGenerationWeightsOwned(
          std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards)), c)
          : vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
    }
    CHECK(w.vision.config == c.raw["vision_config"]);
    CHECK(w.audio.config == c.raw["audio_config"]);
    for (const auto* p : {&w.vision.embedding_projection, &w.audio.embedding_projection}) {
      CHECK(p->dtype == vt::DType::kBF16); CHECK(p->rank == 2); CHECK(p->shape[0] == 2);
    }
    CHECK(w.vision.embedding_projection.shape[1] == 3);
    CHECK(w.audio.embedding_projection.shape[1] == 5);
    REQUIRE(w.vision.embedding_projection.bytes.size() == 12);
    REQUIRE(w.audio.embedding_projection.bytes.size() == 20);
    uint16_t values[6]; std::memcpy(values, w.vision.embedding_projection.bytes.data(), 12);
    for (size_t i = 0; i < 6; ++i) CHECK(values[i] == ((0x3f + i) << 8));
  }
}
TEST_CASE("E4B projectors: null modalities and audio output width fallback") {
  Fixture f; auto c = config(); c.raw["vision_config"] = nullptr; c.raw["audio_config"] = nullptr;
  auto shards = f.open();
  auto w = vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
  CHECK(w.vision.config.is_null()); CHECK(w.audio.embedding_projection.bytes.empty());
  for (const json& output : std::vector<json>{nullptr, 0, 3}) {
    Fixture a; a.add(audio_proj, {2, 3}); auto ac = config();
    ac.raw["audio_config"] = {{"hidden_size", 3}, {"rms_norm_eps", 1e-6}, {"output_proj_dims", output}};
    auto as = a.open(); auto aw = vllm::LoadGemma4ForConditionalGenerationWeights(as, ac);
    CHECK(aw.audio.embedding_projection.shape[1] == 3);
    ac.raw["audio_config"].erase("output_proj_dims");
    CHECK(vllm::LoadGemma4ForConditionalGenerationWeights(as, ac).audio.embedding_projection.bytes.size() == 12);
  }
}
TEST_CASE("E4B projectors: original config dimensions reject tiny projector without allocation") {
  auto original = vllm::LoadHfConfig(E4B_CONFIG_PATH);
  for (const auto* key : {"vision_config", "audio_config"}) {
    auto c = config(); c.raw[key] = original.raw[key]; c.hidden_size = original.hidden_size;
    Fixture f; f.add(std::string(key) == "vision_config" ? vision_proj : audio_proj, {2, 3});
    reject_mm(f, c, "encoder_output");
  }
}
TEST_CASE("E4B projectors: scaled FP8 materialization remains supported") {
  for (const auto& name : {vision_proj, audio_proj}) {
    Fixture f; auto c = config();
    c.raw[name == vision_proj ? "vision_config" : "audio_config"] =
        {{"hidden_size", 3}, {"rms_norm_eps", 1e-6}};
    const size_t start = f.bytes.size(); f.bytes.append(6, '\x38'); // E4M3 1.0
    f.header[name] = {{"dtype", "F8_E4M3"}, {"shape", {2, 3}}, {"data_offsets", {start, start+6}}};
    f.add(name + "_scale", {1}); // BF16 1.0
    auto shards = f.open();
    auto w = vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
    const auto& p = name == vision_proj ? w.vision.embedding_projection : w.audio.embedding_projection;
    REQUIRE(p.bytes.size() == 12); CHECK(p.dtype == vt::DType::kBF16);
    uint16_t values[6]; std::memcpy(values, p.bytes.data(), 12);
    for (auto v : values) CHECK(v == 0x3f80);
  }
}
