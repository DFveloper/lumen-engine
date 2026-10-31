CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 \
cmake -S . -B build -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DVLLM_CPP_CUDA=ON \
  '-DVLLM_CPP_CUDA_ARCHITECTURES=60;61;70' -DVLLM_CPP_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j18
