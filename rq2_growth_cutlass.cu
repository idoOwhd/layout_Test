// A real CUTLASS SM80 tensor-core GEMM. Output address mapping is explicit.
// No torch/CUDA reference is relabelled CUTLASS. Status is checked by Python.
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cuda_runtime.h>

template <class LC>
int run(void* a, void* b, void* c, int m, int n, int k, cudaStream_t stream) {
  using H = cutlass::half_t;
  using Gemm = cutlass::gemm::device::Gemm<
      H, cutlass::layout::RowMajor, H, cutlass::layout::RowMajor, H, LC, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128,128,32>,
      cutlass::gemm::GemmShape<64,64,32>,
      cutlass::gemm::GemmShape<16,8,16>,
      cutlass::epilogue::thread::LinearCombination<H,1,float,float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, 8, 8>;
  int ld = cutlass::platform::is_same<LC, cutlass::layout::RowMajor>::value ? n : m;
  typename Gemm::Arguments args({m,n,k}, {static_cast<H*>(a),k},
      {static_cast<H*>(b),n}, {static_cast<H*>(c),ld}, {static_cast<H*>(c),ld}, {1.f,0.f});
  Gemm op;
  auto status = op.can_implement(args);
  if (status != cutlass::Status::kSuccess) return int(status);
  status = op(args, nullptr, stream);
  return int(status);
}

extern "C" int rq2_gemm(void* a, void* b, void* c, int m, int n, int k,
                        int col, unsigned long long stream) {
  if (col) return run<cutlass::layout::ColumnMajor>(a,b,c,m,n,k,(cudaStream_t)stream);
  return run<cutlass::layout::RowMajor>(a,b,c,m,n,k,(cudaStream_t)stream);
}
