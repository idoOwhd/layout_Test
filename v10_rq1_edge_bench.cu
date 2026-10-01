// v10 L-RQ1 controlled producer -> boundary -> paged-KV consumer benchmark.
// This is a causal CUDA reference, not FlashInfer/vLLM/SGLang runtime evidence.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do { cudaError_t s_ = (expr); if (s_ != cudaSuccess) { \
  std::cerr << "CUDA error: " << cudaGetErrorString(s_) << " at " << __FILE__ \
            << ':' << __LINE__ << '\n'; std::exit(2); } } while (0)

enum Layout : int { NHD = 0, HND = 1 };

struct Case {
  std::string id, stage, subgraph, source, dtype;
  int batch, q_len, kv_len, hq, hkv, dim, page_size, reuse;
};

struct Timing { float median_us, p95_us; };

__device__ __forceinline__ long long cache_offset(
    int layout, int b, int token, int head, int d,
    int pages, int page_size, int heads, int dim) {
  int page = token / page_size;
  int within = token % page_size;
  if (layout == NHD) {
    return (((static_cast<long long>(b) * pages + page) * page_size + within)
             * heads + head) * dim + d;
  }
  return (((static_cast<long long>(b) * pages + page) * heads + head)
           * page_size + within) * dim + d;
}

__device__ __forceinline__ __nv_bfloat16 logical_value(long long logical) {
  return __float2bfloat16(static_cast<float>((logical % 251) - 125) / 128.0f);
}

__global__ void initialize_cache(__nv_bfloat16* cache, int layout, int batch,
                                 int kv_len, int pages, int page_size,
                                 int heads, int dim) {
  long long elements = static_cast<long long>(batch) * kv_len * heads * dim;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += static_cast<long long>(blockDim.x) * gridDim.x) {
    int d = i % dim;
    long long x = i / dim;
    int h = x % heads; x /= heads;
    int t = x % kv_len; int b = x / kv_len;
    cache[cache_offset(layout, b, t, h, d, pages, page_size, heads, dim)] = logical_value(i);
  }
}

__global__ void fill_producer_input(__nv_bfloat16* input, long long elements) {
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += static_cast<long long>(blockDim.x) * gridDim.x) {
    input[i] = logical_value(i + 7919);
  }
}

// Genuine direct emission: each layout receives the logical producer value at
// its own physical address. No transpose is hidden inside this kernel.
__global__ void producer_write(const __nv_bfloat16* input, __nv_bfloat16* cache,
                               int layout, int batch, int q_len, int kv_len,
                               int pages, int page_size, int heads, int dim) {
  long long elements = static_cast<long long>(batch) * q_len * heads * dim;
  int first = kv_len - q_len;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += static_cast<long long>(blockDim.x) * gridDim.x) {
    int d = i % dim;
    long long x = i / dim;
    int h = x % heads; x /= heads;
    int q = x % q_len; int b = x / q_len;
    cache[cache_offset(layout, b, first + q, h, d, pages, page_size, heads, dim)] = input[i];
  }
}

__global__ void convert_cache(const __nv_bfloat16* source, __nv_bfloat16* target,
                              int source_layout, int target_layout, int batch,
                              int kv_len, int pages, int page_size,
                              int heads, int dim) {
  long long elements = static_cast<long long>(batch) * kv_len * heads * dim;
  for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += static_cast<long long>(blockDim.x) * gridDim.x) {
    int d = i % dim;
    long long x = i / dim;
    int h = x % heads; x /= heads;
    int t = x % kv_len; int b = x / kv_len;
    target[cache_offset(target_layout, b, t, h, d, pages, page_size, heads, dim)] =
        source[cache_offset(source_layout, b, t, h, d, pages, page_size, heads, dim)];
  }
}

// Head-local paged decode proxy. HND gives contiguous [within,dim] runs;
// NHD jumps over Hkv for adjacent tokens. Hq/Hkv controls repeated consumers.
__global__ void decode_consumer(const __nv_bfloat16* cache, float* output,
                                int layout, int batch, int kv_len, int pages,
                                int page_size, int hq, int hkv, int dim,
                                int chunks) {
  int chunk = blockIdx.x % chunks;
  int qhead = (blockIdx.x / chunks) % hq;
  int b = blockIdx.x / (chunks * hq);
  int group = hq / hkv;
  int kvhead = qhead / group;
  long long per_head = static_cast<long long>(kv_len) * dim;
  long long lo = per_head * chunk / chunks;
  long long hi = per_head * (chunk + 1) / chunks;
  float sum = 0.0f;
  for (long long i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    int token = i / dim;
    int d = i % dim;
    sum += __bfloat162float(cache[cache_offset(
        layout, b, token, kvhead, d, pages, page_size, hkv, dim)]);
  }
  __shared__ float scratch[256];
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) output[blockIdx.x] = scratch[0];
}

static float quantile(std::vector<float> values, float q) {
  std::sort(values.begin(), values.end());
  return values[static_cast<size_t>(std::llround((values.size() - 1) * q))];
}

static Timing measure(const std::function<void(cudaStream_t)>& launch,
                      int warmup, int iterations) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (int i = 0; i < warmup; ++i) launch(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> values;
  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    launch(stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    values.push_back(ms * 1000.0f);
  }
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return {quantile(values, .5f), quantile(values, .95f)};
}

static std::string cuda_versions() {
  int driver = 0, runtime = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver)); CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  return std::to_string(driver) + "/" + std::to_string(runtime);
}

static void emit(const Case& c, const std::string& run_id,
                 const std::string& strategy, const std::string& producer_layout,
                 const std::string& consumer_layout, const std::string& adaptation,
                 Timing producer, Timing boundary, Timing consumer, Timing edge,
                 long long temp_bytes, long long copied_bytes, bool correct,
                 const std::string& status, const std::string& gpu_name) {
  std::string hypothesis = c.stage == "0-control" ? "H1.NEG" : "H1.1";
  if (c.reuse > 1) hypothesis += "+H1.2";
  if (strategy == "NH-view") hypothesis += "+H1.3";
  if (strategy == "HH-native" || strategy == "NH-copy") hypothesis += "+H1.4";
  auto timing = [&](float value) {
    if (status == "success") std::cout << value;
  };
  std::cout << run_id << ',' << c.id << '-' << strategy << ",L-RQ1," << hypothesis
            << ',' << c.stage << ',' << c.id << ',' << c.subgraph << ',' << c.source
            << ",cuda-reference," << gpu_name << ",," << cuda_versions()
            << ",v10-rq1-cuda-reference," << strategy << ',' << producer_layout
            << ',' << consumer_layout << ',' << adaptation << ',' << c.dtype << ','
            << c.batch << ',' << c.q_len << ',' << c.kv_len << ',' << c.hq << ','
            << c.hkv << ',' << c.page_size << ',' << c.reuse << ',';
  timing(producer.median_us); std::cout << ',';
  timing(boundary.median_us); std::cout << ',';
  timing(consumer.median_us); std::cout << ',';
  timing(edge.median_us); std::cout << ',';
  timing(edge.p95_us);
  std::cout << ',' << temp_bytes << ',' << copied_bytes
            << ",,,,,,," << (correct ? 1 : 0) << ',' << status
            << ",runtime_empirical_controlled\n";
}

static float correctness_error(const __nv_bfloat16* nhd, const __nv_bfloat16* hnd,
                               float* output, const Case& c, int pages, int chunks) {
  int blocks = c.batch * c.hq * chunks;
  decode_consumer<<<blocks, 256>>>(nhd, output, NHD, c.batch, c.kv_len, pages,
                                   c.page_size, c.hq, c.hkv, c.dim, chunks);
  std::vector<float> a(blocks), b(blocks);
  CUDA_CHECK(cudaMemcpy(a.data(), output, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  decode_consumer<<<blocks, 256>>>(hnd, output, HND, c.batch, c.kv_len, pages,
                                   c.page_size, c.hq, c.hkv, c.dim, chunks);
  CUDA_CHECK(cudaMemcpy(b.data(), output, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  float error = 0.0f;
  for (int i = 0; i < blocks; ++i) error = std::max(error, std::abs(a[i] - b[i]));
  return error;
}

static void run_case(const Case& c, const std::string& run_id, int warmup,
                     int iterations, const std::string& gpu_name) {
  const std::vector<std::string> strategy_names = {
      "NN", "NH-copy", "HH-native", "HN-copy", "NH-view"};
  if (c.dtype != "BF16") {
    for (const auto& name : strategy_names) {
      std::string pl = name[0] == 'H' ? "HND" : "NHD";
      std::string cl = (name == "NN" || name == "HN-copy") ? "NHD" : "HND";
      std::string adaptation = name.find("copy") != std::string::npos ? "materialized" :
                               name == "NH-view" ? "view" :
                               name == "HH-native" ? "native-emission" : "none";
      emit(c, run_id, name, pl, cl, adaptation, {}, {}, {}, {}, 0, 0, false,
           "unsupported_native_NVFP4_on_this_reference", gpu_name);
    }
    return;
  }
  int pages = (c.kv_len + c.page_size - 1) / c.page_size;
  long long padded = static_cast<long long>(c.batch) * pages * c.page_size * c.hkv * c.dim;
  long long valid = static_cast<long long>(c.batch) * c.kv_len * c.hkv * c.dim;
  long long produced = static_cast<long long>(c.batch) * c.q_len * c.hkv * c.dim;
  long long cache_bytes = padded * sizeof(__nv_bfloat16);
  long long valid_bytes = valid * sizeof(__nv_bfloat16);
  __nv_bfloat16 *input, *nhd, *hnd, *repair;
  float* output;
  CUDA_CHECK(cudaMalloc(&input, produced * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&nhd, cache_bytes));
  CUDA_CHECK(cudaMalloc(&hnd, cache_bytes));
  CUDA_CHECK(cudaMalloc(&repair, cache_bytes));
  int chunks = 16;
  int consumer_blocks = c.batch * c.hq * chunks;
  CUDA_CHECK(cudaMalloc(&output, consumer_blocks * sizeof(float)));
  int init_blocks = static_cast<int>(std::min<long long>(65535, (valid + 255) / 256));
  int producer_blocks = static_cast<int>(std::min<long long>(65535, (produced + 255) / 256));
  initialize_cache<<<init_blocks, 256>>>(nhd, NHD, c.batch, c.kv_len, pages,
                                         c.page_size, c.hkv, c.dim);
  initialize_cache<<<init_blocks, 256>>>(hnd, HND, c.batch, c.kv_len, pages,
                                         c.page_size, c.hkv, c.dim);
  fill_producer_input<<<producer_blocks, 256>>>(input, produced);
  producer_write<<<producer_blocks, 256>>>(input, nhd, NHD, c.batch, c.q_len,
                                            c.kv_len, pages, c.page_size,
                                            c.hkv, c.dim);
  producer_write<<<producer_blocks, 256>>>(input, hnd, HND, c.batch, c.q_len,
                                            c.kv_len, pages, c.page_size,
                                            c.hkv, c.dim);
  CUDA_CHECK(cudaDeviceSynchronize());
  float max_error = correctness_error(nhd, hnd, output, c, pages, chunks);
  bool correct = max_error <= 1e-3f;

  auto writer = [&](int layout, __nv_bfloat16* target) {
    return [&, layout, target](cudaStream_t stream) {
      producer_write<<<producer_blocks, 256, 0, stream>>>(
          input, target, layout, c.batch, c.q_len, c.kv_len, pages,
          c.page_size, c.hkv, c.dim);
    };
  };
  auto reader = [&](int layout, __nv_bfloat16* source) {
    return [&, layout, source](cudaStream_t stream) {
      for (int r = 0; r < c.reuse; ++r) {
        decode_consumer<<<consumer_blocks, 256, 0, stream>>>(
            source, output, layout, c.batch, c.kv_len, pages, c.page_size,
            c.hq, c.hkv, c.dim, chunks);
      }
    };
  };
  auto converter = [&](int from, int to, __nv_bfloat16* source, __nv_bfloat16* target) {
    return [&, from, to, source, target](cudaStream_t stream) {
      convert_cache<<<init_blocks, 256, 0, stream>>>(
          source, target, from, to, c.batch, c.kv_len, pages, c.page_size,
          c.hkv, c.dim);
    };
  };
  Timing p_nhd = measure(writer(NHD, nhd), warmup, iterations);
  Timing p_hnd = measure(writer(HND, hnd), warmup, iterations);
  Timing c_nhd = measure(reader(NHD, nhd), warmup, iterations);
  Timing c_hnd = measure(reader(HND, hnd), warmup, iterations);
  Timing n2h = measure(converter(NHD, HND, nhd, repair), warmup, iterations);
  Timing h2n = measure(converter(HND, NHD, hnd, repair), warmup, iterations);
  Timing zero{0.0f, 0.0f};

  Timing nn = measure([&](cudaStream_t s) { writer(NHD, nhd)(s); reader(NHD, nhd)(s); }, warmup, iterations);
  Timing nh = measure([&](cudaStream_t s) {
    writer(NHD, nhd)(s); converter(NHD, HND, nhd, repair)(s); reader(HND, repair)(s);
  }, warmup, iterations);
  Timing hh = measure([&](cudaStream_t s) { writer(HND, hnd)(s); reader(HND, hnd)(s); }, warmup, iterations);
  Timing hn = measure([&](cudaStream_t s) {
    writer(HND, hnd)(s); converter(HND, NHD, hnd, repair)(s); reader(NHD, repair)(s);
  }, warmup, iterations);
  // The view path passes NHD strides directly to the stride-polymorphic
  // consumer, so it has no materialization and retains the NHD access cost.
  Timing view = measure([&](cudaStream_t s) { writer(NHD, nhd)(s); reader(NHD, nhd)(s); }, warmup, iterations);

  emit(c, run_id, "NN", "NHD", "NHD", "none", p_nhd, zero, c_nhd, nn,
       0, 0, correct, "success", gpu_name);
  emit(c, run_id, "NH-copy", "NHD", "HND", "materialized", p_nhd, n2h,
       c_hnd, nh, cache_bytes, valid_bytes, correct, "success", gpu_name);
  emit(c, run_id, "HH-native", "HND", "HND", "native-emission", p_hnd, zero,
       c_hnd, hh, 0, 0, correct, "success", gpu_name);
  emit(c, run_id, "HN-copy", "HND", "NHD", "materialized", p_hnd, h2n,
       c_nhd, hn, cache_bytes, valid_bytes, correct, "success", gpu_name);
  emit(c, run_id, "NH-view", "NHD", "stride-polymorphic-HND", "view", p_nhd,
       zero, c_nhd, view, 0, 0, correct, "success", gpu_name);

  CUDA_CHECK(cudaFree(input)); CUDA_CHECK(cudaFree(nhd)); CUDA_CHECK(cudaFree(hnd));
  CUDA_CHECK(cudaFree(repair)); CUDA_CHECK(cudaFree(output));
}

static std::vector<Case> read_cases(const std::string& path) {
  std::ifstream input(path);
  if (!input) { std::cerr << "cannot open " << path << '\n'; std::exit(2); }
  std::string line; std::getline(input, line);
  std::vector<Case> cases;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::stringstream stream(line); std::string field; std::vector<std::string> f;
    while (std::getline(stream, field, '\t')) f.push_back(field);
    if (f.size() != 13) { std::cerr << "bad TSV row: " << line << '\n'; std::exit(2); }
    cases.push_back({f[0], f[1], f[2], f[3], f[4], std::stoi(f[5]),
                     std::stoi(f[6]), std::stoi(f[7]), std::stoi(f[8]),
                     std::stoi(f[9]), std::stoi(f[10]), std::stoi(f[11]),
                     std::stoi(f[12])});
  }
  return cases;
}

int main(int argc, char** argv) {
  std::string cases_path, run_id = "run-0";
  int warmup = 5, iterations = 30;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--cases" && i + 1 < argc) cases_path = argv[++i];
    else if (arg == "--run-id" && i + 1 < argc) run_id = argv[++i];
    else if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (arg == "--iterations" && i + 1 < argc) iterations = std::atoi(argv[++i]);
    else { std::cerr << "usage: " << argv[0] << " --cases TSV --run-id ID [--warmup N --iterations N]\n"; return 2; }
  }
  if (cases_path.empty()) return 2;
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cerr << "run_id=" << run_id << " device=" << prop.name << " cc="
            << prop.major << '.' << prop.minor << '\n';
  std::cout << "run_id,exp_id,rq,hypothesis,stage,case_id,subgraph,source,framework,gpu_name,gpu_uuid,driver_cuda,framework_commit,strategy,layout_producer,layout_consumer,adaptation,dtype,batch,q_len,kv_len,Hq,Hkv,page_size,reuse_count,producer_us,boundary_us,consumer_us,edge_us,edge_p95_us,temp_bytes,bytes_copied,hbm_read_bytes,hbm_write_bytes,l2_hit_rate,producer_winner,edge_winner,edge_regret_pct,correctness_pass,status,evidence_level\n";
  std::cout << std::fixed << std::setprecision(6);
  for (const Case& c : read_cases(cases_path)) {
    if (c.q_len > c.kv_len) {
      std::cerr << "q_len > kv_len for " << c.id << '\n'; return 2;
    }
    run_case(c, run_id, warmup, iterations, prop.name);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
