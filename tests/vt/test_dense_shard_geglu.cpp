// CPU contract for original-format GeGLU shards.
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "vllm/model_executor/layers/linear.h"
#include "vt/backend.h"

int main() {
  using namespace vt;
  Queue q{{DeviceType::kCPU, 0}, nullptr};
  constexpr int K=32, N=3, M=2;
  std::vector<uint16_t> x(M*K);
  for (int i=0;i<M*K;++i) x[i]=F32ToBF16(float(i%7-3)/8);
  Tensor a=Tensor::Contiguous(x.data(),DType::kBF16,q.device,{M,K});
  std::vector<vllm::OwnedTensor> parts(2);
  for(int j=0;j<2;++j) {
    auto& w=parts[j]; w.dtype=j?DType::kQ8_0:DType::kQ4_0;
    w.rank=2; w.shape[0]=N; w.shape[1]=K; w.nk=true;
    w.bytes.resize(N*RowSizeBytes(w.dtype,K));
    for(int r=0;r<N;++r) {
      auto* p=w.bytes.data()+r*RowSizeBytes(w.dtype,K);
      uint16_t scale=F32ToF16(float(r+1)/16); std::memcpy(p,&scale,2);
      for(int k=0;k<(j?32:16);++k) p[2+k]=j?uint8_t(k%9-4):uint8_t((k%8)|((15-k%8)<<4));
    }
  }
  std::vector<uint16_t> gu(M*2*N), expected(M*N);
  std::vector<uint16_t> g(M*N),u(M*N);
  Tensor products[]={Tensor::Contiguous(g.data(),DType::kBF16,q.device,{M,N}),
                     Tensor::Contiguous(u.data(),DType::kBF16,q.device,{M,N})};
  for(int j=0;j<2;++j) { auto w=parts[j].View(); MatmulBTQuant(q,products[j],a,w); }
  for(int m=0;m<M;++m) { std::memcpy(gu.data()+m*2*N,g.data()+m*N,N*2); std::memcpy(gu.data()+m*2*N+N,u.data()+m*N,N*2); }
  Tensor merged=Tensor::Contiguous(gu.data(),DType::kBF16,q.device,{M,2*N});
  Tensor ref=Tensor::Contiguous(expected.data(),DType::kBF16,q.device,{M,N});
  GeluAndMul(q,ref,merged);
  vllm::dense_attn::Dev d{GetBackend(DeviceType::kCPU),q};
  vllm::layers::DenseShardMlpGateUpGeluMethod method(&parts,N);
  auto result=method.Apply(d,a);
  if(std::memcmp(result.ptr(),expected.data(),expected.size()*2)) throw std::runtime_error("mixed GeGLU mismatch");
  Tensor weights[]={parts[0].View(),parts[1].View()};
  bool rejected=false;
  try { MergedGemmDense(q,kDenseGateUpGeluShards,products,a,weights,2,nullptr); } catch(const std::exception&) {rejected=true;}
  if(!rejected) throw std::runtime_error("missing epilogue output accepted");
  auto bad=kDenseGateUpGeluShards;bad.arity=3;rejected=false;
  try { MergedGemmDense(q,bad,products,a,weights,2,&ref); } catch(const std::exception&) {rejected=true;}
  if(!rejected) throw std::runtime_error("bad arity accepted");
  MergedGemmDense(q,kDenseGateUpShards,products,a,weights,2,&ref);
  std::vector<uint16_t> sw(M*N); Tensor swref=Tensor::Contiguous(sw.data(),DType::kBF16,q.device,{M,N});
  MoeSiluMul(q,swref,products[0],products[1]);
  if(std::memcmp(sw.data(),expected.data(),sw.size()*2)) throw std::runtime_error("SwiGLU changed");
  MergedGemmGroup none{2,MergedEpilogue::kNone,kNoMergedFastOp,"none"};
  MergedGemmDense(q,none,products,a,weights,2);
  std::cout<<"mixed Q4/Q8 GeGLU, guards, unchanged SwiGLU and no epilogue PASS\n";
}
