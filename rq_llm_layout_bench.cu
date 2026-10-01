// Controlled CUDA mechanisms for ref_talks RQ1--RQ10.
//
// The benchmark uses shapes from the pinned LLM manifest and holds arithmetic,
// launch geometry and values fixed while changing physical layout or repair
// placement.  It is a reference causal experiment, not an implementation of
// vLLM/SGLang/TensorRT-LLM/FlashInfer.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                          \
  do {                                                                            \
    cudaError_t status_ = (expr);                                                  \
    if (status_ != cudaSuccess) {                                                  \
      std::cerr << "CUDA error " << cudaGetErrorString(status_) << " at "          \
                << __FILE__ << ":" << __LINE__ << std::endl;                       \
      std::exit(2);                                                               \
    }                                                                             \
  } while (0)

enum Layout : int { NHD = 0, HND = 1 };
static int g_order_seed = 0;

struct Case {
  std::string id;
  std::string model;
  std::string phase;
  std::string subgraph;
  std::string attention_kind;
  int tokens;
  int query_heads;
  int heads;
  int dim;
};

struct Timing {
  float p20;
  float p50;
  float p80;
};

__device__ __forceinline__ long long offset_of(int layout, int token, int head,
                                                int dim_index, int tokens,
                                                int heads, int dim) {
  if (layout == NHD) {
    return (static_cast<long long>(token) * heads + head) * dim + dim_index;
  }
  return (static_cast<long long>(head) * tokens + token) * dim + dim_index;
}

__device__ __forceinline__ long long paged_hnd_offset(int token, int head,
                                                       int dim_index, int tokens,
                                                       int heads, int dim,
                                                       int page_size) {
  int page = token / page_size;
  int within = token % page_size;
  return (((static_cast<long long>(page) * heads + head) * page_size + within) * dim
          + dim_index);
}

__global__ void fill_kernel(__half* output, long long elements) {
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += static_cast<long long>(blockDim.x) * gridDim.x) {
    output[i] = __float2half(static_cast<float>((i % 251) - 125) / 128.0f);
  }
}

__global__ void zero_kernel(float* output, long long elements) {
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += static_cast<long long>(blockDim.x) * gridDim.x) {
    output[i] = 0.0f;
  }
}

__global__ void kv_write_kernel(const __half* logical_nhd, __half* physical,
                                int tokens, int heads, int dim, int layout) {
  long long elements = static_cast<long long>(tokens) * heads * dim;
  for (long long logical = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       logical < elements; logical += static_cast<long long>(blockDim.x) * gridDim.x) {
    int d = logical % dim;
    long long tmp = logical / dim;
    int h = tmp % heads;
    int t = tmp / heads;
    physical[offset_of(layout, t, h, d, tokens, heads, dim)] = logical_nhd[logical];
  }
}

__global__ void kv_write_paged_hnd_kernel(const __half* logical_nhd, __half* physical,
                                          int tokens, int heads, int dim,
                                          int page_size, const int* block_table) {
  long long elements = static_cast<long long>(tokens) * heads * dim;
  for (long long logical = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       logical < elements; logical += static_cast<long long>(blockDim.x) * gridDim.x) {
    int d = logical % dim;
    long long tmp = logical / dim;
    int h = tmp % heads;
    int t = tmp / heads;
    int logical_page = t / page_size;
    int physical_page = block_table ? block_table[logical_page] : logical_page;
    int within = t % page_size;
    long long offset = (((static_cast<long long>(physical_page) * heads + h) * page_size
                         + within) * dim + d);
    physical[offset] = logical_nhd[logical];
  }
}

// Head-local consumer: a block works on a contiguous slice of [token, dim] for
// one head.  HND is physically contiguous; NHD jumps over other KV heads.
__global__ void decode_head_scan_kernel(const __half* cache, float* output,
                                        int tokens, int heads, int dim,
                                        int layout, int chunks) {
  int head = blockIdx.x % heads;
  int chunk = blockIdx.x / heads;
  long long per_head = static_cast<long long>(tokens) * dim;
  long long lo = per_head * chunk / chunks;
  long long hi = per_head * (chunk + 1) / chunks;
  float sum = 0.0f;
  for (long long index = lo + threadIdx.x; index < hi; index += blockDim.x) {
    int token = index / dim;
    int d = index % dim;
    sum += __half2float(cache[offset_of(layout, token, head, d, tokens, heads, dim)]);
  }
  __shared__ float scratch[256];
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) output[blockIdx.x] = scratch[0];
}

// Token-local consumer: a block works on [head, dim] for one token.  NHD is
// contiguous; HND jumps over full token ranges.  This approximates a batched
// producer/prefill-side traversal, not a complete attention implementation.
__global__ void token_scan_kernel(const __half* cache, float* output,
                                  int tokens, int heads, int dim, int layout) {
  int token = blockIdx.x;
  float sum = 0.0f;
  long long row = static_cast<long long>(heads) * dim;
  for (long long index = threadIdx.x; index < row; index += blockDim.x) {
    int head = index / dim;
    int d = index % dim;
    sum += __half2float(cache[offset_of(layout, token, head, d, tokens, heads, dim)]);
  }
  __shared__ float scratch[256];
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) output[token] = scratch[0];
}

__global__ void paged_decode_head_kernel(const __half* cache, float* output,
                                         int tokens, int heads, int dim,
                                         int page_size, int chunks,
                                         const int* block_table,
                                         int route_stride) {
  int head = blockIdx.x % heads;
  int chunk = blockIdx.x / heads;
  int selected_tokens = (tokens + route_stride - 1) / route_stride;
  long long per_head = static_cast<long long>(selected_tokens) * dim;
  long long lo = per_head * chunk / chunks;
  long long hi = per_head * (chunk + 1) / chunks;
  float sum = 0.0f;
  for (long long index = lo + threadIdx.x; index < hi; index += blockDim.x) {
    int token = (index / dim) * route_stride;
    int d = index % dim;
    int logical_page = token / page_size;
    int physical_page = block_table[logical_page];
    int within = token % page_size;
    long long offset = (((static_cast<long long>(physical_page) * heads + head) * page_size
                         + within) * dim + d);
    sum += __half2float(cache[offset]);
  }
  __shared__ float scratch[256];
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) output[blockIdx.x] = scratch[0];
}

// Quantized/auxiliary metadata is deliberately represented independently from
// the data.  The four data x metadata combinations let the analyzer measure an
// interaction term instead of assuming that the best data layout and the best
// scale layout can be selected independently.  meta_per_token=false is the
// cache-resident/tiny negative control (one scale per head).
__global__ void scaled_decode_head_scan_kernel(
    const __half* cache, const __half* scales, float* output,
    int tokens, int heads, int dim, int data_layout, int metadata_layout,
    bool meta_per_token, int chunks) {
  int head = blockIdx.x % heads;
  int chunk = blockIdx.x / heads;
  long long per_head = static_cast<long long>(tokens) * dim;
  long long lo = per_head * chunk / chunks;
  long long hi = per_head * (chunk + 1) / chunks;
  float sum = 0.0f;
  for (long long index = lo + threadIdx.x; index < hi; index += blockDim.x) {
    int token = index / dim;
    int d = index % dim;
    long long scale_index = head;
    if (meta_per_token) {
      scale_index = metadata_layout == NHD
          ? static_cast<long long>(token) * heads + head
          : static_cast<long long>(head) * tokens + token;
    }
    float scale = __half2float(scales[scale_index]);
    sum += __half2float(cache[offset_of(data_layout, token, head, d,
                                        tokens, heads, dim)]) * scale;
  }
  __shared__ float scratch[256];
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) output[blockIdx.x] = scratch[0];
}

__global__ void convert_kernel(const __half* source, __half* target,
                               int tokens, int heads, int dim,
                               int source_layout, int target_layout) {
  long long elements = static_cast<long long>(tokens) * heads * dim;
  for (long long logical = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       logical < elements; logical += static_cast<long long>(blockDim.x) * gridDim.x) {
    int d = logical % dim;
    long long tmp = logical / dim;
    int h = tmp % heads;
    int t = tmp / heads;
    target[offset_of(target_layout, t, h, d, tokens, heads, dim)] =
        source[offset_of(source_layout, t, h, d, tokens, heads, dim)];
  }
}

__device__ __forceinline__ __half rope_like(__half value, int dim_index) {
  float x = __half2float(value);
  float scale = (dim_index & 1) ? 0.999f : 1.001f;
  return __float2half(x * scale + 0.03125f);
}

__global__ void rope_kernel(const __half* source, __half* target,
                            int tokens, int heads, int dim, int layout) {
  long long elements = static_cast<long long>(tokens) * heads * dim;
  for (long long physical = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       physical < elements; physical += static_cast<long long>(blockDim.x) * gridDim.x) {
    int d = physical % dim;
    target[physical] = rope_like(source[physical], d);
  }
}

__global__ void fused_rope_convert_kernel(const __half* source_nhd, __half* target_hnd,
                                          int tokens, int heads, int dim) {
  long long elements = static_cast<long long>(tokens) * heads * dim;
  for (long long logical = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       logical < elements; logical += static_cast<long long>(blockDim.x) * gridDim.x) {
    int d = logical % dim;
    long long tmp = logical / dim;
    int h = tmp % heads;
    int t = tmp / heads;
    target_hnd[offset_of(HND, t, h, d, tokens, heads, dim)] = rope_like(source_nhd[logical], d);
  }
}

__global__ void pressure_kernel(const __half* source, __half* target, long long elements) {
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += static_cast<long long>(blockDim.x) * gridDim.x) {
    float x = __half2float(source[i]);
    target[i] = __float2half(x * 1.0001f + 0.0001f);
  }
}

static float quantile(std::vector<float> values, float fraction) {
  std::sort(values.begin(), values.end());
  size_t index = static_cast<size_t>(std::llround((values.size() - 1) * fraction));
  return values[index];
}

static Timing measure(const std::function<void(cudaStream_t)>& launch,
                      int warmup, int iterations, cudaStream_t stream = nullptr) {
  for (int i = 0; i < warmup; ++i) launch(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> values;
  values.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    launch(stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    values.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return {quantile(values, .2f), quantile(values, .5f), quantile(values, .8f)};
}

static Timing measure_contended(const std::function<void(cudaStream_t)>& foreground,
                                const std::function<void(cudaStream_t)>& pressure,
                                int warmup, int iterations) {
  cudaStream_t foreground_stream, pressure_stream, control_stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&foreground_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&pressure_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&control_stream, cudaStreamNonBlocking));
  auto once = [&](float* elapsed) {
    cudaEvent_t start, done;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&done));
    CUDA_CHECK(cudaEventRecord(start, control_stream));
    CUDA_CHECK(cudaStreamWaitEvent(foreground_stream, start));
    CUDA_CHECK(cudaStreamWaitEvent(pressure_stream, start));
    pressure(pressure_stream);
    foreground(foreground_stream);
    CUDA_CHECK(cudaEventRecord(done, foreground_stream));
    CUDA_CHECK(cudaEventSynchronize(done));
    if (elapsed) CUDA_CHECK(cudaEventElapsedTime(elapsed, start, done));
    CUDA_CHECK(cudaStreamSynchronize(pressure_stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(done));
  };
  for (int i = 0; i < warmup; ++i) once(nullptr);
  std::vector<float> values;
  for (int i = 0; i < iterations; ++i) {
    float elapsed = 0.0f;
    once(&elapsed);
    values.push_back(elapsed);
  }
  CUDA_CHECK(cudaStreamDestroy(foreground_stream));
  CUDA_CHECK(cudaStreamDestroy(pressure_stream));
  CUDA_CHECK(cudaStreamDestroy(control_stream));
  return {quantile(values, .2f), quantile(values, .5f), quantile(values, .8f)};
}

static void emit(const Case& c, const char* benchmark, const std::string& strategy,
                 const std::string& layout, Timing timing, long long bytes,
                 float max_error, int page_size = 0, int reuse_count = 0,
                 int fanout_head = 0, int fanout_token = 0,
                 int allocated_tokens = 0, const std::string& trace_kind = "") {
  std::cout << "cuda-reference," << benchmark << ',' << c.id << ',' << c.subgraph << ',' << c.phase
            << ',' << strategy << ',' << layout << ',' << c.tokens << ',' << c.heads
            << ',' << c.query_heads << ',' << c.dim << ',' << page_size << ',' << reuse_count << ',' << bytes << ',' << timing.p20
            << ',' << timing.p50 << ',' << timing.p80 << ',' << max_error << ',' << c.model
            << "," << c.attention_kind << ",runtime_empirical," << fanout_head << ','
            << fanout_token << ',' << allocated_tokens << ',' << trace_kind << ','
            << g_order_seed << "\n";
}

static float compare_buffers(const __half* a, const __half* b, long long elements) {
  std::vector<__half> host_a(elements), host_b(elements);
  CUDA_CHECK(cudaMemcpy(host_a.data(), a, elements * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_b.data(), b, elements * sizeof(__half), cudaMemcpyDeviceToHost));
  float error = 0.0f;
  for (long long i = 0; i < elements; ++i) {
    error = std::max(error, std::abs(__half2float(host_a[i]) - __half2float(host_b[i])));
  }
  return error;
}

static float compare_float_buffers(const float* a, const float* b, long long elements) {
  std::vector<float> host_a(elements), host_b(elements);
  CUDA_CHECK(cudaMemcpy(host_a.data(), a, elements * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_b.data(), b, elements * sizeof(float), cudaMemcpyDeviceToHost));
  float error = 0.0f;
  for (long long i = 0; i < elements; ++i) {
    error = std::max(error, std::abs(host_a[i] - host_b[i]));
  }
  return error;
}

static void run_case(const Case& c, int warmup, int iterations) {
  long long elements = static_cast<long long>(c.tokens) * c.heads * c.dim;
  long long bytes = elements * sizeof(__half);
  __half *logical, *nhd, *hnd, *scratch, *reference;
  float *reductions, *reference_reductions;
  CUDA_CHECK(cudaMalloc(&logical, bytes));
  CUDA_CHECK(cudaMalloc(&nhd, bytes));
  CUDA_CHECK(cudaMalloc(&hnd, bytes));
  CUDA_CHECK(cudaMalloc(&scratch, bytes));
  CUDA_CHECK(cudaMalloc(&reference, bytes));
  int chunks = 32;
  CUDA_CHECK(cudaMalloc(&reductions, std::max(c.tokens, c.heads * chunks) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&reference_reductions,
                        std::max(c.tokens, c.heads * chunks) * sizeof(float)));
  int blocks = static_cast<int>(std::min<long long>(65535, (elements + 255) / 256));
  fill_kernel<<<blocks, 256>>>(logical, elements);
  kv_write_kernel<<<blocks, 256>>>(logical, nhd, c.tokens, c.heads, c.dim, NHD);
  kv_write_kernel<<<blocks, 256>>>(logical, hnd, c.tokens, c.heads, c.dim, HND);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> layout_order = {NHD, HND};
  if (g_order_seed & 1) std::reverse(layout_order.begin(), layout_order.end());
  for (int layout : layout_order) {
    __half* cache = layout == NHD ? nhd : hnd;
    const char* name = layout == NHD ? "NHD" : "HND";
    auto write = measure([&](cudaStream_t stream) {
      kv_write_kernel<<<blocks, 256, 0, stream>>>(logical, cache, c.tokens, c.heads, c.dim, layout);
    }, warmup, iterations);
    kv_write_kernel<<<blocks, 256>>>(logical, cache, c.tokens, c.heads, c.dim, layout);
    if (layout == HND) {
      convert_kernel<<<blocks, 256>>>(cache, scratch, c.tokens, c.heads, c.dim, HND, NHD);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    float write_error = compare_buffers(layout == NHD ? cache : scratch, logical, elements);
    emit(c, "kv_write", name, name, write, bytes, write_error);

    auto decode = measure([&](cudaStream_t stream) {
      decode_head_scan_kernel<<<c.heads * chunks, 256, 0, stream>>>(
          cache, reductions, c.tokens, c.heads, c.dim, layout, chunks);
    }, warmup, iterations);
    decode_head_scan_kernel<<<c.heads * chunks, 256>>>(
        cache, reductions, c.tokens, c.heads, c.dim, layout, chunks);
    decode_head_scan_kernel<<<c.heads * chunks, 256>>>(
        nhd, reference_reductions, c.tokens, c.heads, c.dim, NHD, chunks);
    CUDA_CHECK(cudaDeviceSynchronize());
    float decode_error = compare_float_buffers(
        reductions, reference_reductions, static_cast<long long>(c.heads) * chunks);
    emit(c, "decode_head_scan", name, name, decode, bytes, decode_error);

    auto token_scan = measure([&](cudaStream_t stream) {
      token_scan_kernel<<<c.tokens, 256, 0, stream>>>(cache, reductions, c.tokens, c.heads, c.dim, layout);
    }, warmup, iterations);
    token_scan_kernel<<<c.tokens, 256>>>(
        cache, reductions, c.tokens, c.heads, c.dim, layout);
    token_scan_kernel<<<c.tokens, 256>>>(
        nhd, reference_reductions, c.tokens, c.heads, c.dim, NHD);
    CUDA_CHECK(cudaDeviceSynchronize());
    float token_error = compare_float_buffers(reductions, reference_reductions, c.tokens);
    emit(c, "token_major_scan", name, name, token_scan, bytes, token_error);
  }

  auto n2h = measure([&](cudaStream_t stream) {
    convert_kernel<<<blocks, 256, 0, stream>>>(nhd, scratch, c.tokens, c.heads, c.dim, NHD, HND);
  }, warmup, iterations);
  convert_kernel<<<blocks, 256>>>(nhd, scratch, c.tokens, c.heads, c.dim, NHD, HND);
  CUDA_CHECK(cudaDeviceSynchronize());
  float n2h_error = compare_buffers(scratch, hnd, elements);
  emit(c, "layout_conversion", "NHD_to_HND", "NHD->HND", n2h, bytes * 2, n2h_error);
  auto h2n = measure([&](cudaStream_t stream) {
    convert_kernel<<<blocks, 256, 0, stream>>>(hnd, scratch, c.tokens, c.heads, c.dim, HND, NHD);
  }, warmup, iterations);
  convert_kernel<<<blocks, 256>>>(hnd, scratch, c.tokens, c.heads, c.dim, HND, NHD);
  CUDA_CHECK(cudaDeviceSynchronize());
  float h2n_error = compare_buffers(scratch, logical, elements);
  emit(c, "layout_conversion", "HND_to_NHD", "HND->NHD", h2n, bytes * 2, h2n_error);

  // Directly time conversion amortization rather than assuming primitive
  // latencies add perfectly.  All strategies begin from the same logical NHD
  // producer output and perform the same number of head-local reads.
  // Dense around the expected crossover; four logarithmic points were too
  // sparse to distinguish a real threshold from one noisy winning sample.
  for (int reuse : {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}) {
    auto common_nhd = measure([&](cudaStream_t stream) {
      kv_write_kernel<<<blocks, 256, 0, stream>>>(logical, nhd, c.tokens, c.heads, c.dim, NHD);
      for (int r = 0; r < reuse; ++r) {
        decode_head_scan_kernel<<<c.heads * chunks, 256, 0, stream>>>(
            nhd, reductions, c.tokens, c.heads, c.dim, NHD, chunks);
      }
    }, warmup, iterations);
    emit(c, "conversion_reuse_pipeline", "common_NHD", "NHD", common_nhd,
         bytes * (1 + reuse), 0.0f, 0, reuse);
    auto direct_hnd = measure([&](cudaStream_t stream) {
      kv_write_kernel<<<blocks, 256, 0, stream>>>(logical, hnd, c.tokens, c.heads, c.dim, HND);
      for (int r = 0; r < reuse; ++r) {
        decode_head_scan_kernel<<<c.heads * chunks, 256, 0, stream>>>(
            hnd, reductions, c.tokens, c.heads, c.dim, HND, chunks);
      }
    }, warmup, iterations);
    emit(c, "conversion_reuse_pipeline", "producer_native_HND", "HND", direct_hnd,
         bytes * (1 + reuse), 0.0f, 0, reuse);
    auto convert_once = measure([&](cudaStream_t stream) {
      kv_write_kernel<<<blocks, 256, 0, stream>>>(logical, nhd, c.tokens, c.heads, c.dim, NHD);
      convert_kernel<<<blocks, 256, 0, stream>>>(nhd, hnd, c.tokens, c.heads, c.dim, NHD, HND);
      for (int r = 0; r < reuse; ++r) {
        decode_head_scan_kernel<<<c.heads * chunks, 256, 0, stream>>>(
            hnd, reductions, c.tokens, c.heads, c.dim, HND, chunks);
      }
    }, warmup, iterations);
    emit(c, "conversion_reuse_pipeline", "NHD_then_convert_once_HND", "NHD->HND",
         convert_once, bytes * (3 + reuse), 0.0f, 0, reuse);
  }

  // RQ2: directly vary consumer multiplicity.  A single 1:1 pair cannot show
  // when a common layout, a converted split, or a different common layout is
  // optimal.  Every row times the complete producer -> all-consumers path.
  std::vector<std::pair<int, int>> fanouts = {
      {1, 8}, {1, 4}, {1, 2}, {1, 1}, {2, 1}, {4, 1}, {8, 1}};
  if (g_order_seed & 1) std::reverse(fanouts.begin(), fanouts.end());
  for (const auto& fanout : fanouts) {
    int head_fanout = fanout.first;
    int token_fanout = fanout.second;
    auto multi_nhd = measure([&](cudaStream_t stream) {
      kv_write_kernel<<<blocks, 256, 0, stream>>>(logical, nhd, c.tokens, c.heads, c.dim, NHD);
      for (int i = 0; i < head_fanout; ++i)
        decode_head_scan_kernel<<<c.heads * chunks, 256, 0, stream>>>(
            nhd, reductions, c.tokens, c.heads, c.dim, NHD, chunks);
      for (int i = 0; i < token_fanout; ++i)
        token_scan_kernel<<<c.tokens, 256, 0, stream>>>(
            nhd, reductions, c.tokens, c.heads, c.dim, NHD);
    }, warmup, iterations);
    emit(c, "weighted_multi_consumer_pipeline", "common_NHD", "NHD", multi_nhd,
         bytes * (1 + head_fanout + token_fanout), 0.0f, 0, 0,
         head_fanout, token_fanout);
    if (head_fanout == 1 && token_fanout == 1)
      emit(c, "multi_consumer_pipeline", "common_NHD", "NHD", multi_nhd,
           bytes * 3, 0.0f);
    auto multi_hnd = measure([&](cudaStream_t stream) {
      kv_write_kernel<<<blocks, 256, 0, stream>>>(logical, hnd, c.tokens, c.heads, c.dim, HND);
      for (int i = 0; i < head_fanout; ++i)
        decode_head_scan_kernel<<<c.heads * chunks, 256, 0, stream>>>(
            hnd, reductions, c.tokens, c.heads, c.dim, HND, chunks);
      for (int i = 0; i < token_fanout; ++i)
        token_scan_kernel<<<c.tokens, 256, 0, stream>>>(
            hnd, reductions, c.tokens, c.heads, c.dim, HND);
    }, warmup, iterations);
    emit(c, "weighted_multi_consumer_pipeline", "common_HND", "HND", multi_hnd,
         bytes * (1 + head_fanout + token_fanout), 0.0f, 0, 0,
         head_fanout, token_fanout);
    if (head_fanout == 1 && token_fanout == 1)
      emit(c, "multi_consumer_pipeline", "common_HND", "HND", multi_hnd,
           bytes * 3, 0.0f);
    auto multi_split = measure([&](cudaStream_t stream) {
      kv_write_kernel<<<blocks, 256, 0, stream>>>(logical, nhd, c.tokens, c.heads, c.dim, NHD);
      convert_kernel<<<blocks, 256, 0, stream>>>(nhd, hnd, c.tokens, c.heads, c.dim, NHD, HND);
      for (int i = 0; i < head_fanout; ++i)
        decode_head_scan_kernel<<<c.heads * chunks, 256, 0, stream>>>(
            hnd, reductions, c.tokens, c.heads, c.dim, HND, chunks);
      for (int i = 0; i < token_fanout; ++i)
        token_scan_kernel<<<c.tokens, 256, 0, stream>>>(
            nhd, reductions, c.tokens, c.heads, c.dim, NHD);
    }, warmup, iterations);
    emit(c, "weighted_multi_consumer_pipeline", "split_NHD_HND_with_conversion", "NHD+HND",
         multi_split, bytes * (3 + head_fanout + token_fanout), 0.0f, 0, 0,
         head_fanout, token_fanout);
    if (head_fanout == 1 && token_fanout == 1)
      emit(c, "multi_consumer_pipeline", "split_NHD_HND_with_conversion", "NHD+HND",
           multi_split, bytes * 5, 0.0f);
  }

  // RQ6: persistent-state policies on measured access traces.  Earlier
  // evidence multiplied primitive medians in a cost model; these rows launch
  // the conversions and consumers themselves.  0=token-local, 1=head-local.
  const std::vector<std::pair<std::string, std::vector<int>>> traces = {
      {"stationary_token", std::vector<int>(32, 0)},
      {"stationary_head", std::vector<int>(32, 1)},
      {"alternating", {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
                        0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1}},
      {"phase_drift", {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}},
      {"bursty", {0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,
                   0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1}},
      {"seeded_markov", {0,0,1,1,1,0,0,0,1,1,0,0,1,1,1,1,
                          1,0,0,1,0,0,0,1,1,1,0,1,1,0,0,0}},
  };
  std::vector<std::pair<std::string, int>> policies = {
      {"fixed_NHD", -1}, {"fixed_HND", -2}, {"eager_switch", 1},
      {"hysteresis_2", 2}, {"hysteresis_4", 4}};
  if (g_order_seed & 1) std::reverse(policies.begin(), policies.end());
  for (const auto& trace_item : traces) {
    const std::string& trace_name = trace_item.first;
    const std::vector<int>& accesses = trace_item.second;
    for (const auto& policy_item : policies) {
      const std::string& policy_name = policy_item.first;
      int threshold = policy_item.second;
      auto launch_policy = [&](cudaStream_t stream) {
        int current = threshold == -2 ? HND : NHD;
        __half* current_buffer = current == NHD ? nhd : hnd;
        kv_write_kernel<<<blocks, 256, 0, stream>>>(
            logical, current_buffer, c.tokens, c.heads, c.dim, current);
        int streak = 0;
        for (int access : accesses) {
          int desired = access ? HND : NHD;
          if (threshold > 0 && desired != current) {
            ++streak;
            if (streak >= threshold) {
              __half* source = current == NHD ? nhd : hnd;
              __half* target = desired == NHD ? nhd : hnd;
              convert_kernel<<<blocks, 256, 0, stream>>>(
                  source, target, c.tokens, c.heads, c.dim, current, desired);
              current = desired;
              streak = 0;
            }
          } else if (desired == current) {
            streak = 0;
          }
          __half* cache = current == NHD ? nhd : hnd;
          if (access) {
            decode_head_scan_kernel<<<c.heads * chunks, 256, 0, stream>>>(
                cache, reductions, c.tokens, c.heads, c.dim, current, chunks);
          } else {
            token_scan_kernel<<<c.tokens, 256, 0, stream>>>(
                cache, reductions, c.tokens, c.heads, c.dim, current);
          }
        }
      };
      Timing policy_time = measure(launch_policy, warmup, iterations);
      launch_policy(nullptr);
      int final_access = accesses.back();
      if (final_access) {
        decode_head_scan_kernel<<<c.heads * chunks, 256>>>(
            hnd, reference_reductions, c.tokens, c.heads, c.dim, HND, chunks);
      } else {
        token_scan_kernel<<<c.tokens, 256>>>(
            nhd, reference_reductions, c.tokens, c.heads, c.dim, NHD);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      long long compared = final_access ? c.heads * chunks : c.tokens;
      float policy_error = compare_float_buffers(reductions, reference_reductions, compared);
      emit(c, "state_migration_trace", policy_name, "persistent_dynamic", policy_time,
           bytes * (1 + static_cast<long long>(accesses.size())), policy_error,
           0, static_cast<int>(accesses.size()), 0, 0, c.tokens, trace_name);
    }
  }

  auto separate = measure([&](cudaStream_t stream) {
    rope_kernel<<<blocks, 256, 0, stream>>>(nhd, scratch, c.tokens, c.heads, c.dim, NHD);
    convert_kernel<<<blocks, 256, 0, stream>>>(scratch, reference, c.tokens, c.heads, c.dim, NHD, HND);
  }, warmup, iterations);
  rope_kernel<<<blocks, 256>>>(nhd, scratch, c.tokens, c.heads, c.dim, NHD);
  convert_kernel<<<blocks, 256>>>(scratch, reference, c.tokens, c.heads, c.dim, NHD, HND);
  CUDA_CHECK(cudaDeviceSynchronize());
  emit(c, "rope_kv_pipeline", "rope_then_convert", "NHD->HND", separate, bytes * 4, 0.0f);

  auto convert_then_rope = measure([&](cudaStream_t stream) {
    convert_kernel<<<blocks, 256, 0, stream>>>(nhd, scratch, c.tokens, c.heads, c.dim, NHD, HND);
    rope_kernel<<<blocks, 256, 0, stream>>>(scratch, hnd, c.tokens, c.heads, c.dim, HND);
  }, warmup, iterations);
  convert_kernel<<<blocks, 256>>>(nhd, scratch, c.tokens, c.heads, c.dim, NHD, HND);
  rope_kernel<<<blocks, 256>>>(scratch, hnd, c.tokens, c.heads, c.dim, HND);
  CUDA_CHECK(cudaDeviceSynchronize());
  float order_error = compare_buffers(reference, hnd, elements);
  emit(c, "rope_kv_pipeline", "convert_then_rope", "NHD->HND", convert_then_rope,
       bytes * 4, order_error);

  auto fused = measure([&](cudaStream_t stream) {
    fused_rope_convert_kernel<<<blocks, 256, 0, stream>>>(nhd, hnd, c.tokens, c.heads, c.dim);
  }, warmup, iterations);
  fused_rope_convert_kernel<<<blocks, 256>>>(nhd, hnd, c.tokens, c.heads, c.dim);
  CUDA_CHECK(cudaDeviceSynchronize());
  float fused_error = compare_buffers(reference, hnd, elements);
  emit(c, "rope_kv_pipeline", "fused_rope_native_HND", "NHD->HND", fused, bytes * 2, fused_error);

  // RQ8: factorial data-layout x metadata-layout experiment plus a small,
  // cache-resident metadata negative control.  Every candidate performs the
  // same arithmetic and reads the same logical values.
  __half *token_scales_nhd, *token_scales_hnd, *head_scales;
  long long token_scale_elements = static_cast<long long>(c.tokens) * c.heads;
  CUDA_CHECK(cudaMalloc(&token_scales_nhd, token_scale_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&token_scales_hnd, token_scale_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&head_scales, c.heads * sizeof(__half)));
  fill_kernel<<<std::min(65535, (c.tokens * c.heads + 255) / 256), 256>>>(
      token_scales_nhd, token_scale_elements);
  // Preserve identical logical scale[token, head] values in both physical
  // layouts.  Reinterpreting one flat allocation with two index functions
  // changes the mathematical input and invalidates a paired layout test.
  int token_scale_blocks = static_cast<int>(
      std::min<long long>(65535, (token_scale_elements + 255) / 256));
  convert_kernel<<<token_scale_blocks, 256>>>(
      token_scales_nhd, token_scales_hnd, c.tokens, c.heads, 1, NHD, HND);
  fill_kernel<<<1, 256>>>(head_scales, c.heads);
  CUDA_CHECK(cudaDeviceSynchronize());
  for (bool per_token : {false, true}) {
    for (int data_layout : layout_order) {
      for (int metadata_layout : layout_order) {
        __half* cache = data_layout == NHD ? nhd : hnd;
        __half* scales = per_token
            ? (metadata_layout == NHD ? token_scales_nhd : token_scales_hnd)
            : head_scales;
        std::string strategy = std::string("data_") + (data_layout == NHD ? "NHD" : "HND")
            + "_meta_" + (metadata_layout == NHD ? "NHD" : "HND")
            + (per_token ? "_per_token" : "_per_head");
        auto timing = measure([&](cudaStream_t stream) {
          scaled_decode_head_scan_kernel<<<c.heads * chunks, 256, 0, stream>>>(
              cache, scales, reductions, c.tokens, c.heads, c.dim, data_layout,
              metadata_layout, per_token, chunks);
        }, warmup, iterations);
        // Compare with the canonical NHD data + NHD metadata representation.
        // The candidate is timed alone; these launches are correctness-only.
        scaled_decode_head_scan_kernel<<<c.heads * chunks, 256>>>(
            cache, scales, reductions, c.tokens, c.heads, c.dim, data_layout,
            metadata_layout, per_token, chunks);
        scaled_decode_head_scan_kernel<<<c.heads * chunks, 256>>>(
            nhd, per_token ? token_scales_nhd : head_scales,
            reference_reductions, c.tokens, c.heads, c.dim, NHD, NHD,
            per_token, chunks);
        CUDA_CHECK(cudaDeviceSynchronize());
        float metadata_error = compare_float_buffers(
            reductions, reference_reductions, static_cast<long long>(c.heads) * chunks);
        long long metadata_bytes = (per_token ? static_cast<long long>(c.tokens) * c.heads
                                               : c.heads) * sizeof(__half);
        emit(c, "data_metadata_scan", strategy,
             data_layout == NHD ? "NHD" : "HND", timing,
             bytes + metadata_bytes, metadata_error, 0, per_token ? 1 : 0);
      }
    }
  }
  CUDA_CHECK(cudaFree(token_scales_nhd));
  CUDA_CHECK(cudaFree(token_scales_hnd));
  CUDA_CHECK(cudaFree(head_scales));

  // RQ8/RQ10: actual padded pages and actual logical->physical block-table
  // permutations.  Non-divisible live-token counts are deliberately retained;
  // they were previously skipped, which made fragmentation conclusions purely
  // formula-derived.
  for (int page_size : {1, 4, 8, 16, 32, 64, 128, 256}) {
    int pages = (c.tokens + page_size - 1) / page_size;
    int allocated_tokens = pages * page_size;
    long long paged_elements = static_cast<long long>(allocated_tokens) * c.heads * c.dim;
    __half* paged_cache;
    CUDA_CHECK(cudaMalloc(&paged_cache, paged_elements * sizeof(__half)));
    std::vector<std::pair<std::string, std::vector<int>>> tables;
    std::vector<int> identity(pages), reverse(pages), permuted(pages);
    std::iota(identity.begin(), identity.end(), 0);
    reverse = identity;
    std::reverse(reverse.begin(), reverse.end());
    // Deterministic bijection with no random-library/version dependency.
    for (int i = 0; i < pages; ++i) permuted[i] = (static_cast<long long>(i) * 17 + 3) % pages;
    std::vector<int> seen(pages, 0);
    bool bijective = true;
    for (int value : permuted) {
      if (seen[value]++) bijective = false;
    }
    if (!bijective) {
      permuted = identity;
      for (int i = 0; i + 1 < pages; i += 2) std::swap(permuted[i], permuted[i + 1]);
    }
    tables.push_back({"identity", identity});
    tables.push_back({"reverse", reverse});
    tables.push_back({"permuted", permuted});
    if (g_order_seed & 1) std::reverse(tables.begin(), tables.end());
    for (const auto& table : tables) {
      int* block_table;
      CUDA_CHECK(cudaMalloc(&block_table, pages * sizeof(int)));
      CUDA_CHECK(cudaMemcpy(block_table, table.second.data(), pages * sizeof(int),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(paged_cache, 0, paged_elements * sizeof(__half)));
      kv_write_paged_hnd_kernel<<<blocks, 256>>>(logical, paged_cache, c.tokens, c.heads,
                                                 c.dim, page_size, block_table);
      CUDA_CHECK(cudaDeviceSynchronize());
      for (int route_stride : {1, 4, 16}) {
        auto paged = measure([&](cudaStream_t stream) {
          paged_decode_head_kernel<<<c.heads * chunks, 256, 0, stream>>>(
              paged_cache, reductions, c.tokens, c.heads, c.dim, page_size, chunks,
              block_table, route_stride);
        }, warmup, iterations);
        std::string strategy = "page_" + std::to_string(page_size) + "_route_"
            + std::to_string(route_stride) + "_table_" + table.first;
        emit(c, route_stride == 1 ? "paged_decode_head" : "sparse_paged_decode_head",
             strategy, "paged_HND", paged,
             paged_elements * sizeof(__half) + pages * sizeof(int), 0.0f,
             page_size, route_stride, 0, 0, allocated_tokens, table.first);
      }
      CUDA_CHECK(cudaFree(block_table));
    }
    CUDA_CHECK(cudaFree(paged_cache));
  }

  // HBM/copy pressure is a single-GPU proxy for RQ4.  It must not be reported
  // as an All2All or multi-rank experiment.
  long long pressure_elements = std::max<long long>(elements, 32LL * 1024 * 1024);
  __half *pressure_in, *pressure_out;
  CUDA_CHECK(cudaMalloc(&pressure_in, pressure_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&pressure_out, pressure_elements * sizeof(__half)));
  fill_kernel<<<65535, 256>>>(pressure_in, pressure_elements);
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int layout : layout_order) {
    __half* cache = layout == NHD ? nhd : hnd;
    const char* name = layout == NHD ? "NHD" : "HND";
    auto foreground = [&](cudaStream_t stream) {
      decode_head_scan_kernel<<<c.heads * chunks, 256, 0, stream>>>(
          cache, reductions, c.tokens, c.heads, c.dim, layout, chunks);
    };
    auto pressure = [&](cudaStream_t stream) {
      for (int repeat = 0; repeat < 4; ++repeat) {
        pressure_kernel<<<65535, 256, 0, stream>>>(pressure_in, pressure_out, pressure_elements);
      }
    };
    Timing contended = measure_contended(foreground, pressure, std::max(1, warmup / 4),
                                         std::max(5, iterations / 4));
    emit(c, "decode_under_hbm_contention", name, name, contended, bytes, 0.0f);
  }
  CUDA_CHECK(cudaFree(pressure_in));
  CUDA_CHECK(cudaFree(pressure_out));
  CUDA_CHECK(cudaFree(logical));
  CUDA_CHECK(cudaFree(nhd));
  CUDA_CHECK(cudaFree(hnd));
  CUDA_CHECK(cudaFree(scratch));
  CUDA_CHECK(cudaFree(reference));
  CUDA_CHECK(cudaFree(reductions));
  CUDA_CHECK(cudaFree(reference_reductions));
}

int main(int argc, char** argv) {
  bool quick = false;
  int warmup = 10;
  int iterations = 40;
  std::string case_file;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--quick") quick = true;
    else if (arg == "--case-file" && i + 1 < argc) case_file = argv[++i];
    else if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (arg == "--iterations" && i + 1 < argc) iterations = std::atoi(argv[++i]);
    else if (arg == "--order-seed" && i + 1 < argc) g_order_seed = std::atoi(argv[++i]);
    else {
      std::cerr << "usage: " << argv[0] << " [--quick] [--case-file TSV] [--warmup N] [--iterations N] [--order-seed N]\n";
      return 2;
    }
  }
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cerr << "device=" << prop.name << " cc=" << prop.major << '.' << prop.minor << '\n';
  std::cout << "framework,benchmark,case_id,subgraph,phase,strategy,layout,tokens,kv_heads,query_heads,head_dim,page_size,reuse_count,bytes,p20_ms,p50_ms,p80_ms,max_abs_error,model,attention_kind,evidence_level,fanout_head,fanout_token,allocated_tokens,trace_kind,order_seed\n";
  std::vector<Case> cases = {
      {"decode_qwen3_0.6b_gqa", "Qwen/Qwen3-0.6B@c1899de", "decode", "gqa", "GQA", 8192, 16, 8, 128},
      {"prefill_qwen3_0.6b_gqa", "Qwen/Qwen3-0.6B@c1899de", "prefill", "gqa", "GQA", 2048, 16, 8, 128},
      {"decode_llama_gqa", "Llama-family-GQA-policy-shape", "decode", "gqa", "GQA", 4096, 32, 8, 128},
      {"decode_mqa", "MQA-policy-shape", "decode", "gqa", "MQA", 8192, 32, 1, 128},
      {"decode_mha", "MHA-policy-shape", "decode", "gqa", "MHA", 4096, 32, 32, 128},
  };
  if (!case_file.empty()) {
    std::ifstream input(case_file);
    if (!input) {
      std::cerr << "cannot open case file: " << case_file << '\n';
      return 2;
    }
    cases.clear();
    std::string line;
    std::getline(input, line);  // header
    while (std::getline(input, line)) {
      if (line.empty()) continue;
      std::vector<std::string> fields;
      std::stringstream stream(line);
      std::string field;
      while (std::getline(stream, field, '\t')) fields.push_back(field);
      if (fields.size() != 9) {
        std::cerr << "invalid TSV row: " << line << '\n';
        return 2;
      }
      cases.push_back({fields[0], fields[1], fields[2], fields[3], fields[4],
                       std::stoi(fields[5]), std::stoi(fields[6]),
                       std::stoi(fields[7]), std::stoi(fields[8])});
    }
  }
  if (quick) {
    // Generated quick manifests are already stratified.  Truncating them here
    // would silently discard subgraphs and MHA/MQA/GQA shape classes.
    if (case_file.empty() && cases.size() > 2) cases.resize(2);
    warmup = std::min(warmup, 3);
    iterations = std::min(iterations, 8);
  }
  if (!cases.empty()) {
    size_t shift = static_cast<size_t>(std::abs(g_order_seed)) % cases.size();
    std::rotate(cases.begin(), cases.begin() + shift, cases.end());
    if (g_order_seed & 1) std::reverse(cases.begin(), cases.end());
  }
  for (const Case& c : cases) run_case(c, warmup, iterations);
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
