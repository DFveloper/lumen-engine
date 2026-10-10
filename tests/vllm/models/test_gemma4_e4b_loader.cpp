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
  CHECK(w.lm_head.shape[0] == 2); CHECK(w.lm_head.shape[1] == 3);
  REQUIRE(w.lm_head.bytes.size() == 12);
  uint16_t actual[6]; std::memcpy(actual, w.lm_head.bytes.data(), sizeof(actual));
  const uint16_t expected[] = {0x3f80, 0x4040, 0x40a0, 0x4000, 0x4080, 0x40c0};
  for (size_t i = 0; i < 6; ++i) CHECK(actual[i] == expected[i]);
  std::vector<vllm::SafetensorsFile> second;
  second.push_back(vllm::SafetensorsFile::Open(f.path.string()));
  auto owned = std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(second));
  auto ow = vllm::LoadGemma4ForConditionalGenerationWeightsOwned(owned, c);
  CHECK(ow.lm_head.bytes.size() == w.lm_head.bytes.size());
  CHECK(std::memcmp(ow.lm_head.bytes.data(), w.lm_head.bytes.data(), 12) == 0);
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
