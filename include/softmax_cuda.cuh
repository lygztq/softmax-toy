#include "cuda_runtime.h"
#include "tensor.h"
#include <algorithm>
#include <limits>

template <typename Ty> __device__ Ty exp(Ty val) { return exp(val); }

template <> __device__ float exp(float val) { return __expf(val); }

template <typename Ty, typename Func>
__inline__ __device__ Ty warp_reduce(Ty val, Ty init_val, Func func) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    auto tmp = __shfl_down_sync(0xffffffff, val, offset);
    if (threadIdx.x + offset >= blockDim.x)
      tmp = init_val;
    val = func(val, tmp);
  }
  return val;
}

template <typename Ty, typename Func>
__inline__ __device__ Ty block_reduce(Ty val, Ty init_val, Func func) {
  auto lane_id = threadIdx.x % warpSize;
  auto warp_id = threadIdx.x / warpSize;
  static __shared__ Ty tmp[32];

  val = warp_reduce(val, init_val, func);

  if (lane_id == 0) {
    tmp[warp_id] = val;
  }
  __syncthreads();

  val = (threadIdx.x * warpSize < blockDim.x) ? tmp[lane_id] : init_val;
  if (warp_id == 0) {
    val = warp_reduce(val, init_val, func);
  }
  return val;
}

template <typename Ty>
__global__ void softmax_kernel(const Ty *__restrict__ input,
                               Ty *__restrict__ output, int64_t outer_dim,
                               int64_t inner_dim, int64_t reduce_dim,
                               int64_t nelem) {
  int64_t vector_id = blockIdx.x;
  int64_t vth_id = threadIdx.x;
  int64_t num_vth = blockDim.x;
  int64_t num_reduce_inner_loop = (reduce_dim + num_vth - 1) / num_vth;
  __shared__ Ty global_max;
  __shared__ Ty global_sum;

  int64_t outer = vector_id / inner_dim;
  int64_t inner = vector_id % inner_dim;
  int64_t base = outer * inner_dim * reduce_dim + inner;

  Ty local_max = std::numeric_limits<Ty>::lowest();
  Ty local_sum = Ty(0);

  // each thread grep its own max value
  for (int64_t i = 0; i < num_reduce_inner_loop; ++i) {
    int64_t idx = base + (vth_id + i * num_vth) * inner_dim;
    if (idx >= nelem)
      break;
    Ty curr = input[idx];
    local_max = (local_max < curr) ? curr : local_max;
  }
  __syncthreads();

  local_max = block_reduce(local_max, std::numeric_limits<Ty>::lowest(),
                           [](Ty x, Ty y) { return (x > y) ? x : y; });

  if (threadIdx.x == 0) {
    global_max = local_max;
  }
  __syncthreads();

  local_max = global_max;
  for (int64_t i = 0; i < num_reduce_inner_loop; ++i) {
    int64_t idx = base + (vth_id + i * num_vth) * inner_dim;
    if (idx >= nelem)
      break;
    output[idx] = exp<Ty>(input[idx] - local_max);
    local_sum += output[idx];
  }
  __syncthreads();
  local_sum = block_reduce(local_sum, Ty(0), [](Ty x, Ty y) { return x + y; });

  if (threadIdx.x == 0) {
    global_sum = local_sum;
  }
  __syncthreads();

  local_sum = global_sum;
  for (int64_t i = 0; i < num_reduce_inner_loop; ++i) {
    int64_t idx = base + (vth_id + i * num_vth) * inner_dim;
    if (idx >= nelem)
      break;
    output[idx] /= local_sum;
  }
}

template <typename Ty>
CudaTensor<Ty> softmax_cuda(const CudaTensor<Ty> &input, int64_t axis) {
  CudaTensor<Ty> out_tensor = CudaTensor<Ty>::cuda_tensor(input.shape);
  auto multiply = [](int64_t a, int64_t b) { return a * b; };
  int64_t outer_dim = std::reduce(
      out_tensor.shape.begin(), out_tensor.shape.begin() + axis, 1LL, multiply);
  int64_t reduce_dim = input.shape[axis];
  int64_t inner_dim = std::reduce(out_tensor.shape.begin() + axis + 1,
                                  out_tensor.shape.end(), 1LL, multiply);

  Ty *out_ptr = out_tensor.ptr();
  const Ty *in_ptr = input.ptr();

  dim3 block(std::min(1024LL, reduce_dim));
  dim3 grid(inner_dim * outer_dim);
  softmax_kernel<Ty><<<grid, block>>>(in_ptr, out_ptr, outer_dim, inner_dim,
                                      reduce_dim, input.nelem());
  return std::move(out_tensor);
}
