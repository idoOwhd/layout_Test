#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(2); }} while (0)

template <int Mode>
__global__ void bank_kernel(float *out, int repeats) {
  extern __shared__ float storage[];
  volatile float *tile = storage;
  int lane = threadIdx.x;
  int stride = Mode == 1 ? 33 : 32;
  for (int col = 0; col < 32; ++col) {
    int physical_col = Mode == 2 ? (col ^ lane) : col;
    tile[lane * stride + physical_col] = float(lane + col) * 0.001f;
  }
  __syncthreads();
  float sum = 0.0f;
  for (int r = 0; r < repeats; ++r) {
    int logical_col = r & 31;
    int physical_col = Mode == 2 ? (logical_col ^ lane) : logical_col;
    sum += tile[lane * stride + physical_col];
  }
  out[blockIdx.x * 32 + lane] = sum;
}

template <int Mode>
float measure(float *out, int blocks, int repeats, int warmup, int iterations) {
  size_t smem = size_t(32 * (Mode == 1 ? 33 : 32)) * sizeof(float);
  for (int i = 0; i < warmup; ++i) bank_kernel<Mode><<<blocks, 32, smem>>>(out, repeats);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> times;
  for (int i = 0; i < iterations; ++i) {
    cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a)); bank_kernel<Mode><<<blocks, 32, smem>>>(out, repeats);
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); times.push_back(ms);
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
  }
  std::sort(times.begin(), times.end()); return times[times.size() / 2];
}

int main(int argc, char **argv) {
  int blocks = 4096, repeats = 4096, warmup = 10, iterations = 50;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--blocks")) blocks = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--repeats")) repeats = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--warmup")) warmup = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--iterations")) iterations = std::atoi(argv[i + 1]);
  }
  float *out = nullptr; CUDA_CHECK(cudaMalloc(&out, size_t(blocks) * 32 * sizeof(float)));
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("framework,case_id,layout,blocks,repeats,p50_ms,device_name,compute_capability\n");
  std::printf("cuda,shared_transpose_tile,row_major_stride32,%d,%d,%.6f,%s,%d.%d\n", blocks, repeats, measure<0>(out, blocks, repeats, warmup, iterations), prop.name, prop.major, prop.minor);
  std::printf("cuda,shared_transpose_tile,padded_stride33,%d,%d,%.6f,%s,%d.%d\n", blocks, repeats, measure<1>(out, blocks, repeats, warmup, iterations), prop.name, prop.major, prop.minor);
  std::printf("cuda,shared_transpose_tile,xor_swizzle,%d,%d,%.6f,%s,%d.%d\n", blocks, repeats, measure<2>(out, blocks, repeats, warmup, iterations), prop.name, prop.major, prop.minor);
  CUDA_CHECK(cudaFree(out)); return 0;
}
