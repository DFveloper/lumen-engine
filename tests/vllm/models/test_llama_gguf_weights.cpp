// Llama GGUF adapter regression: pinned llama.cpp 62318cd57 conversion/llama.py:175-181.
#include <doctest/doctest.h>
#include <cstring>
#include "gguf_builder.h"
#include "vllm/model_executor/models/llama_gguf_weights.h"
using namespace gguf_test;
namespace {
std::string Model(bool tied, bool scaled = false, bool qnorm = false, bool quant = false, std::string refusal = "") {
  const int h=quant?32:16;
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "llama"));
  for (auto [key,value] : std::vector<std::pair<std::string,uint32_t>>{
      {"embedding_length",static_cast<uint32_t>(h)},{"block_count",1},{"attention.head_count",2},
      {"attention.head_count_kv",1},{"feed_forward_length",static_cast<uint32_t>(2*h)},{"context_length",256},
      {"rope.dimension_count",static_cast<uint32_t>(h/2)}}) b.AddKv(U32Kv("llama."+key,value));
  b.AddKv(F32Kv("llama.attention.layer_norm_rms_epsilon",1e-5F));
  b.AddKv(F32Kv("llama.rope.freq_base",10000.F));
  if(scaled) b.AddKv(StrKv("llama.rope.scaling.type","linear"));
  if(refusal=="partial") b.AddKv(U32Kv("llama.attention.value_length",3));
  auto add=[&](std::string name,int rows,int cols,float base=0.F) {
    const bool projection = quant && (name.find("attn_q.weight")!=std::string::npos ||
        name.find("attn_k.weight")!=std::string::npos || name.find("attn_v.weight")!=std::string::npos ||
        name.find("ffn_gate.weight")!=std::string::npos || name.find("ffn_up.weight")!=std::string::npos);
    if(projection) {
      const bool q8=name.find("attn_k")!=std::string::npos || name.find("ffn_up")!=std::string::npos;
      const int bytes=q8?34:18;
      std::string packed(static_cast<size_t>(rows*bytes),'\0');
      for(int r=0;r<rows;++r) {
        packed[r*bytes+1]=0x3c;  // f16 scale 1; payload identifies complete rows.
        for(int c=2;c<bytes;++c)packed[r*bytes+c]=static_cast<char>(r+c);
      }
      b.AddTensor(name,{static_cast<uint64_t>(cols),static_cast<uint64_t>(rows)},q8?8:2,packed);
      return;
    }
    std::string data(static_cast<size_t>(rows*cols*4),'\0');
    for(int r=0;r<rows;++r) for(int c=0;c<cols;++c) {
      float x=base+r; std::memcpy(data.data()+(r*cols+c)*4,&x,4);
    }
    b.AddTensor(name,cols==1?std::vector<uint64_t>{static_cast<uint64_t>(rows)}:
                std::vector<uint64_t>{static_cast<uint64_t>(cols),static_cast<uint64_t>(rows)},0,data);
  };
  add("token_embd.weight",32,h); add("output_norm.weight",h,1);
  if(!tied) add("output.weight",32,h);
  add("blk.0.attn_norm.weight",h,1);add("blk.0.ffn_norm.weight",h,1);
  add("blk.0.attn_q.weight",h,h,0);add("blk.0.attn_k.weight",h/2,h,32);
  add("blk.0.attn_v.weight",h/2,h,64);add("blk.0.attn_output.weight",h,h);
  add("blk.0.ffn_gate.weight",2*h,h);add("blk.0.ffn_up.weight",2*h,h);
  add("blk.0.ffn_down.weight",h,2*h);
  if(qnorm)add("blk.0.attn_q_norm.weight",h/2,1);
  if(refusal=="bias")add("blk.0.attn_q.bias",h,1);
  if(refusal=="rope")add("rope_freqs.weight",h/4,1);
  return b.Build();
}
}
TEST_CASE("Llama GGUF config maps GQA and tied output without QK normalization") {
  for(bool tied:{false,true}) {
    TempFile f(Model(tied));auto g=vllm::GgufFile::Open(f.path());
    auto cfg=vllm::LlamaHfConfigFromGguf(g);
    CHECK(cfg.architectures==std::vector<std::string>{"LlamaForCausalLM"});
    CHECK(cfg.num_attention_heads==2);CHECK(cfg.num_key_value_heads==1);
    CHECK(cfg.head_dim==8);CHECK(cfg.vocab_size==32);CHECK(cfg.rotary_dim==8);
    CHECK(cfg.raw.value("tie_word_embeddings",!tied)==tied);
    const vllm::GgufLoadPolicy pol;
    auto w=vllm::LoadLlamaFromGguf(g,cfg,&pol);
    REQUIRE(w.layers.size()==1);CHECK(w.layers[0].attn.q_norm.Empty());
    CHECK(w.layers[0].attn.k_norm.Empty());
    CHECK(w.tie_word_embeddings==tied);
    const auto& parts=w.layers[0].attn.qkv_proj_parts;
    REQUIRE(parts.size()==3);REQUIRE(parts[0].dtype==vt::DType::kBF16);
    REQUIRE(parts[0].shape[0]==16);
    const auto* p=reinterpret_cast<const uint16_t*>(parts[0].bytes.data());
    for(int h=0;h<2;++h)for(int i=0;i<8;++i)
      CHECK(p[(h*8+i)*16]==vt::F32ToBF16(float(h*8+2*(i%4)+i/4)));
    p=reinterpret_cast<const uint16_t*>(parts[1].bytes.data());
    for(int i=0;i<8;++i)CHECK(p[i*16]==vt::F32ToBF16(float(32+2*(i%4)+i/4)));
    p=reinterpret_cast<const uint16_t*>(parts[2].bytes.data());
    for(int i=0;i<8;++i)CHECK(p[i*16]==vt::F32ToBF16(float(64+i)));
  }
}
TEST_CASE("Llama GGUF refuses unimplemented scaled RoPE and QK norm variants") {
  for(auto variants:{std::pair{true,false},std::pair{false,true}}) {
    TempFile f(Model(false,variants.first,variants.second));auto g=vllm::GgufFile::Open(f.path());
    CHECK_THROWS_AS(vllm::LlamaHfConfigFromGguf(g),std::runtime_error);
  }
}

TEST_CASE("Llama GGUF preserves heterogeneous quant blocks through QK row inversion") {
  TempFile f(Model(false,false,false,true));auto g=vllm::GgufFile::Open(f.path());
  auto cfg=vllm::LlamaHfConfigFromGguf(g);
  vllm::GgufLoadPolicy pol;pol.keep_quant=true;pol.expand_nk=true;
  auto w=vllm::LoadLlamaFromGguf(g,cfg,&pol);
  const auto& parts=w.layers[0].attn.qkv_proj_parts;
  REQUIRE(parts.size()==3);CHECK(parts[0].dtype==vt::DType::kQ4_0);
  CHECK(parts[1].dtype==vt::DType::kQ8_0);CHECK(parts[2].dtype==vt::DType::kQ4_0);
  REQUIRE(w.layers[0].mlp.gate_up_proj_parts.size()==2);
  CHECK(w.layers[0].mlp.gate_up_proj_parts[0].dtype==vt::DType::kQ4_0);
  CHECK(w.layers[0].mlp.gate_up_proj_parts[1].dtype==vt::DType::kQ8_0);
  for(int part=0;part<3;++part) {
    const auto& p=parts[part]; const int bytes=part==1?34:18;
    const std::string name=part==0?"attn_q":part==1?"attn_k":"attn_v";
    const auto& source=g.Get("blk.0."+name+".weight");
    CHECK(p.bytes.size()==static_cast<size_t>(p.shape[0]*bytes));
    for(int row=0;row<p.shape[0];++row) {
      const int from=part==2?row:(row/16)*16+2*(row%8)+(row%16)/8;
      CHECK(std::memcmp(p.bytes.data()+row*bytes,source.data+from*bytes,bytes)==0);
    }
  }
}

TEST_CASE("Llama GGUF refuses bias, sidecar RoPE, and asymmetric value heads") {
  for(const auto* name:{"bias","rope","partial"}) {
    TempFile f(Model(false,false,false,false,name));auto g=vllm::GgufFile::Open(f.path());
    CHECK_THROWS_AS(vllm::LlamaHfConfigFromGguf(g),std::runtime_error);
  }
}
