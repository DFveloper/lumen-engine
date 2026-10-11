#include <doctest/doctest.h>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include "vllm/model_executor/models/gemma4_loaded.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/multimodal/gemma4_image_processor.h"

namespace {
using nlohmann::json;
json golden() {
  std::ifstream f(std::string(E4B_VISION_FIXTURES)+"/oracle.json");
  return json::parse(f);
}
vllm::HfConfig config(const json& j) {
  vllm::HfConfig c;
  c.hidden_size=2; c.vocab_size=3; c.num_hidden_layers=1;
  c.num_attention_heads=1; c.num_key_value_heads=1; c.head_dim=2;
  c.raw={{"text_config",json::object()},{"vision_config",j.at("config")}};
  return c;
}
vllm::Gemma4Weights weights(const vllm::HfConfig& c) {
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(std::string(E4B_VISION_FIXTURES)+"/model.safetensors"));
  return vllm::LoadGemma4ForConditionalGenerationWeights(shards,c);
}
void compare(const std::vector<float>& actual,const json& expected,float tolerance) {
  const auto ref=expected.get<std::vector<float>>(); REQUIRE(actual.size()==ref.size());
  float error=0; for(size_t i=0;i<ref.size();++i) {
    REQUIRE(std::isfinite(actual[i]));
    error=std::max(error,std::abs(actual[i]-ref[i]));
  }
  INFO("max absolute error="<<error); CHECK(error<=tolerance);
}
}
TEST_CASE("E4B vision: independent nonzero BF16 Torch stages") {
  const auto j=golden(); auto c=config(j); auto w=weights(c);
  auto cfg=*w.vision_config; auto vw=w.vision_patch_weights;
  const auto& connector=w.vision.embedding_projection;
  vw.embed_projection.resize(connector.bytes.size()/2);
  std::memcpy(vw.embed_projection.data(),connector.bytes.data(),connector.bytes.size());
  const auto& small=j.at("small");
  auto pixels=small.at("pixels").get<std::vector<float>>();
  auto pos=small.at("positions").get<std::vector<int64_t>>();
  vllm::multimodal::Gemma4VisionCapture capture;
  auto& backend=vt::GetBackend(vt::DeviceType::kCPU);
  auto out=vllm::multimodal::Gemma4VisionForward(pixels,pos,18,vw,cfg,backend,&capture);
  compare(capture.patch_embed_out,small.at("patch"),0.004f);
  compare(capture.attention_out,small.at("attention"),0.032f);
  compare(capture.mlp_out,small.at("mlp"),0.032f);
  compare(capture.encoder_out,small.at("encoder"),0.032f);
  compare(capture.pooled,small.at("pooled"),0.032f);
  compare(out,small.at("projected"),0.008f);
  for (int mode=0; mode<6; ++mode) {
    auto bad=cfg;
    if(mode==0)bad.num_kv_heads=2;
    if(mode==1)bad.hidden_size=5;
    if(mode==2)bad.head_dim=2;
    if(mode==3)bad.rms_norm_eps=0;
    if(mode==4)bad.rope_theta=0;
    if(mode==5)bad.num_heads=INT64_MAX;
    CHECK_THROWS_WITH(vllm::multimodal::Gemma4VisionForward(pixels,pos,18,vw,bad,backend),
                      doctest::Contains("execution config"));
  }
  auto broken=vw; broken.blocks[0].k_proj.pop_back();
  CHECK_THROWS_WITH(vllm::multimodal::Gemma4VisionForward(pixels,pos,18,broken,cfg,backend),
                    doctest::Contains("weight shape"));
  auto badpos=pos;badpos[2]=0;
  CHECK_THROWS(vllm::multimodal::Gemma4VisionForward(pixels,badpos,18,vw,cfg,backend));
  pixels[0]=std::numeric_limits<float>::quiet_NaN();
  CHECK_THROWS(vllm::multimodal::Gemma4VisionForward(pixels,pos,18,vw,cfg,backend));

}
TEST_CASE("E4B vision: decoded RGB reaches loaded shared registry and PLE merge") {
  auto j=golden(); auto c=config(j); auto w=weights(c);
  const vllm::ModelRegistration reg{"Gemma4ForConditionalGeneration",&vllm::kGemma4MmHooks,{}};
  vllm::Gemma4LoadedModel model(reg,std::move(w));
  std::vector<uint8_t> rgb(384*384*3);
  for(size_t i=0;i<rgb.size();++i)rgb[i]=static_cast<uint8_t>((i*17+i/7)%256);
  auto features=vllm::multimodal::Gemma4ImageProcessor(70).ProcessImageRGB(rgb,384,384);
  vllm::multimodal::MultiModalFeatureSpec item;
  item.modality="image";item.length=static_cast<int>(features.num_soft_tokens);
  item.data=std::make_shared<vllm::multimodal::ImageKwargs>(std::move(features.patches));
  vt::Queue q{};
  auto out=vllm::ModelRegistry::EncodeMm(model,c,q,item);
  REQUIRE(out.embeds.dtype==vt::DType::kF32); REQUIRE(out.embeds.shape[0]==item.length);
  REQUIRE(out.embeds.shape[1]==2); REQUIRE(out.storage);
  compare(std::vector<float>(out.embeds.Ptr<float>(),out.embeds.Ptr<float>()+item.length*2),j.at("image_projected"),0.016f);
  std::vector<int32_t> ids(static_cast<size_t>(item.length)+2,1); ids.front()=2;
  std::vector<char> mask(ids.size(),1);mask.front()=mask.back()=0;
  std::vector<vt::Tensor> slices{out.embeds};
  vllm::MmEmbedInputs in;in.token_ids=&ids;in.is_mm_embed=&mask;in.mm_embeds=&slices;
  auto merged=vllm::ModelRegistry::EmbedMm(model,c,q,in);
  for(int i=0;i<item.length;++i) {
    CHECK(merged.mm.ple_token_ids.Ptr<int32_t>()[i+1]==0);
    for(int h=0;h<2;++h)CHECK(merged.mm.inputs_embeds.Ptr<uint16_t>()[(i+1)*2+h]==vt::F32ToBF16(out.embeds.Ptr<float>()[i*2+h]));
  }
  CHECK(merged.mm.ple_token_ids.Ptr<int32_t>()[0]==2);
  CHECK(merged.mm.ple_token_ids.Ptr<int32_t>()[ids.size()-1]==1);
  CHECK_FALSE(vllm::ModelRegistry::SupportsMmInputs(model));
  auto original=*item.data;
  for(int mode=0;mode<8;++mode) {
    if(mode==0)item.data->image_grid_thw[0]=2;
    if(mode==1)item.data->image_grid_thw[1]=INT64_MAX;
    if(mode==2)item.data->image_grid_thw[2]=0;
    if(mode==3)item.data->pixel_values_bf16.push_back(0);
    if(mode==4)item.data->pixel_values_f32.pop_back();
    if(mode==5)item.data->pixel_values_f32.back()=1;
    if(mode==6)item.data->patch_feature_dim=4;
    if(mode==7)item.data->num_patches=INT64_MAX;
    CHECK_THROWS(vllm::ModelRegistry::EncodeMm(model,c,q,item));
    *item.data=original;
  }
  item.length++; CHECK_THROWS(vllm::ModelRegistry::EncodeMm(model,c,q,item));item.length--;
  item.modality="video";CHECK_THROWS(vllm::ModelRegistry::EncodeMm(model,c,q,item));item.modality="image";
  q.device.type=vt::DeviceType::kCUDA;CHECK_THROWS(vllm::ModelRegistry::EncodeMm(model,c,q,item));
}
