// llama-cpp 62318cd57: mmq.cuh:85-92, vecdotq.cuh:650-675.
// Independent raw Q6_K scalar contraction through the registered MatmulBT op.
#include <doctest/doctest.h>
#ifdef VLLM_CPP_CUDA
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "pascal_q6_prefill_fixture.h"

namespace {
bool Q6PrefillGpu() {
#ifdef VLLM_CPP_CUDA
  try { vt::GetBackend(vt::DeviceType::kCUDA); } catch (...) { return false; }
  cudaDeviceProp p{};
  REQUIRE(cudaGetDeviceProperties(&p, 0) == cudaSuccess);
  return p.major == 6 && p.minor >= 1 && p.sharedMemPerBlock >= 48 * 1024;
#else
  return false;
#endif
}
vt::Tensor Q6Tensor(void* data, vt::DType dtype, int64_t rows, int64_t cols) {
  vt::Tensor t;
  t.data=data; t.dtype=dtype; t.device={vt::DeviceType::kCUDA,0};
  t.rank=2; t.shape[0]=rows; t.shape[1]=cols; t.stride[0]=cols; t.stride[1]=1;
  return t;
}
int RawQ6(const uint8_t* b, int j) {
  const int half=j/128, quarter=(j%128)/32, lane=j%32;
  const int lo=(b[half*64+(quarter%2)*32+lane] >> (quarter>=2 ? 4 : 0)) & 15;
  const int hi=(b[128+half*32+lane] >> (2*quarter)) & 3;
  return (lo | (hi<<4))-32;
}
float Half(const uint8_t* p) { uint16_t h; std::memcpy(&h,p,2); return vt::F16ToF32(h); }
struct Q6Expected { float value; double absolute_terms; };
Q6Expected RawD4(const std::vector<float>& a, const std::vector<uint8_t>& w,
                 int64_t row, int64_t col, int64_t k) {
  float result=0;
  double absolute=0;
  for(int64_t b=0;b<k/256;++b) {
    const uint8_t* weight=w.data()+(col*(k/256)+b)*210;
    float subtotal=0;
    for(int group=0;group<8;++group) {
      const float* x=a.data()+row*k+b*256+group*32;
      float amax=0;
      for(int j=0;j<32;++j) amax=std::max(amax,std::abs(x[j]));
      const float inv=amax==0 ? 0 : 127.0f/amax;
      const float d=amax==0 ? 0 : 1.0f/inv;
      int integer=0;
      for(int j=0;j<32;++j) {
        const int index=group*32+j;
        const int quant=static_cast<int>(std::round(x[j]*inv));
        const int scale=static_cast<int8_t>(weight[192+index/16]);
        const int term=quant*RawQ6(weight,index)*scale;
        integer+=term;
        absolute+=std::abs(double(term)*d*Half(weight+208));
      }
      subtotal+=d*integer;
    }
    result+=Half(weight+208)*subtotal;
  }
  return {result,absolute};
}
std::vector<float> Q6Run(std::vector<float> a,const std::vector<uint8_t>& w,
                         int64_t m,int64_t n,int64_t k,vt::DType input,vt::DType output) {
  auto& gpu=vt::GetBackend(vt::DeviceType::kCUDA);
  auto q=gpu.CreateQueue();
  std::vector<uint16_t> ab(a.size());
  if(input==vt::DType::kBF16)
    for(size_t i=0;i<a.size();++i) { ab[i]=vt::F32ToBF16(a[i]); a[i]=vt::BF16ToF32(ab[i]); }
  const size_t asize=a.size()*(input==vt::DType::kF32?4:2);
  const size_t osize=m*n*(output==vt::DType::kF32?4:2);
  void* da=gpu.Alloc(asize); void* dw=gpu.Alloc(w.size()); void* dout=gpu.Alloc(osize);
  gpu.Copy(q,da,input==vt::DType::kF32?static_cast<void*>(a.data()):static_cast<void*>(ab.data()),asize);
  gpu.Copy(q,dw,w.data(),w.size());
  auto at=Q6Tensor(da,input,m,k),wt=Q6Tensor(dw,vt::DType::kQ6_K,n,k),ot=Q6Tensor(dout,output,m,n);
  vt::MatmulBT(q,ot,at,wt);
  std::vector<float> actual(m*n);
  std::vector<uint16_t> ob(m*n);
  gpu.Copy(q,output==vt::DType::kF32?static_cast<void*>(actual.data()):static_cast<void*>(ob.data()),dout,osize);
  gpu.Synchronize(q);
  gpu.Free(da); gpu.Free(dw); gpu.Free(dout); gpu.DestroyQueue(q);
  if(output==vt::DType::kBF16)
    for(size_t i=0;i<actual.size();++i) actual[i]=vt::BF16ToF32(ob[i]);
  return actual;
}
void Q6Compare(std::vector<float> a,const std::vector<uint8_t>& w,
                int64_t m,int64_t n,int64_t k,vt::DType input,vt::DType output) {
  const auto actual=Q6Run(a,w,m,n,k,input,output);
  if(input==vt::DType::kBF16)
    for(float& x:a) x=vt::BF16ToF32(vt::F32ToBF16(x));
  for(int64_t row=0;row<m;++row) for(int64_t col=0;col<n;++col) {
    const auto expected=RawD4(a,w,row,col,k);
    const float rounded=output==vt::DType::kBF16 ? vt::BF16ToF32(vt::F32ToBF16(expected.value)) : expected.value;
    // Summation roundoff scales with absolute term magnitudes. A BF16 result
    // permits one output ULP when a reduction order crosses its midpoint.
    double bound=16*std::numeric_limits<float>::epsilon()*expected.absolute_terms;
    if(output==vt::DType::kBF16) bound+=std::abs(rounded)/128.0;
    CHECK(std::isfinite(actual[row*n+col]));
    CHECK(std::abs(double(actual[row*n+col])-rounded)<=bound);
  }
}
}  // namespace

TEST_CASE("Pascal Q6_K MMQ prefill reproduces actual oracle projection input") {
  if(!Q6PrefillGpu()) return;
  using namespace pascal_q6_prefill_fixture;
  std::vector<float> a;
  for(int i=0;i<16;++i) a.insert(a.end(),kInput.begin(),kInput.end());
  const std::vector<uint8_t> w(kWeight.begin(),kWeight.end());
  Q6Compare(a,w,16,3,1536,vt::DType::kF32,vt::DType::kF32);
  for(int col=0;col<3;++col)
    CHECK(RawD4(a,w,0,col,1536).value==doctest::Approx(kOracleDown[col]).epsilon(2e-5).scale(1));
}

TEST_CASE("Pascal Q6_K D4 prefill preserves signed packed weights and F32 scales") {
  if(!Q6PrefillGpu()) return;
  for(int64_t m:{9,16,128}) for(int64_t k:{256,512,1536}) {
    const int64_t n=m==9?1:(m==16?7:17);
    std::vector<uint8_t> w(n*(k/256)*210);
    for(size_t i=0;i<w.size();++i) w[i]=static_cast<uint8_t>((i*73+19)^(i>>3));
    for(int64_t b=0;b<n*(k/256);++b) {
      const uint16_t d=vt::F32ToF16(.00390625f);
      std::memcpy(w.data()+b*210+208,&d,2);
      for(int s=0;s<16;++s) w[b*210+192+s]=static_cast<uint8_t>(s%5==0?0:(s%2?-128:127));
    }
    for(int kind=0;kind<3;++kind) {
      CAPTURE(m);CAPTURE(n);CAPTURE(k);CAPTURE(kind);
      std::vector<float> a(m*k);
      const float magnitudes[]={1e-6f,.001003f,.13007f,1.0003f,2.0007f,8.001f,.05003f,100.003f};
      for(int64_t j=0;j<m*k;++j)
        a[j]=kind==0?0:std::sin(float(j)*.37f)*(kind==2?1e-6f:magnitudes[(j/32)%8]);
      for(auto input:{vt::DType::kF32,vt::DType::kBF16})
        for(auto output:{vt::DType::kF32,vt::DType::kBF16})
          Q6Compare(a,w,m,n,k,input,output);
    }
  }
}

TEST_CASE("Pascal Q6_K MMQ leaves the M8 boundary on its existing contraction") {
  if(!Q6PrefillGpu()) return;
  constexpr int64_t m=8,n=3,k=512;
  std::vector<float> a(m*k);
  for(size_t i=0;i<a.size();++i) a[i]=std::cos(float(i)*.173f)*(1+(i/32)%5);
  std::vector<uint8_t> w(n*(k/256)*210);
  for(size_t i=0;i<w.size();++i) w[i]=static_cast<uint8_t>(i*51+(i>>4));
  for(int64_t b=0;b<n*k/256;++b) {
    const uint16_t d=vt::F32ToF16(.00390625f);
    std::memcpy(w.data()+b*210+208,&d,2);
  }
  std::vector<float> baseline(m*n);
  const vt::Device cpu{vt::DeviceType::kCPU,0};
  vt::Queue q{cpu,nullptr};
  auto at=vt::Tensor::Contiguous(a.data(),vt::DType::kF32,cpu,{m,k});
  auto wt=vt::Tensor::Contiguous(w.data(),vt::DType::kF32,cpu,{n,k});wt.dtype=vt::DType::kQ6_K;
  auto ot=vt::Tensor::Contiguous(baseline.data(),vt::DType::kF32,cpu,{m,n});
  vt::MatmulBT(q,ot,at,wt);
  const auto actual=Q6Run(a,w,m,n,k,vt::DType::kF32,vt::DType::kF32);
  for(size_t i=0;i<actual.size();++i)
    CHECK(actual[i]==doctest::Approx(baseline[i]).epsilon(2e-5).scale(1));
}
