#include "tensor.h"
#include <cmath>

template <typename Ty>
CpuTensor<Ty> softmax_cpu(const CpuTensor<Ty> &input, int64_t axis) {
  CpuTensor<Ty> out_tensor = CpuTensor<Ty>::cpu_tensor(input.shape);
  auto multiply = [](int64_t a, int64_t b) { return a * b; };
  int64_t outer_dim = std::reduce(
      out_tensor.shape.begin(), out_tensor.shape.begin() + axis, 1LL, multiply);
  int64_t reduce_dim = input.shape[axis];
  int64_t inner_dim = std::reduce(out_tensor.shape.begin() + axis + 1,
                                  out_tensor.shape.end(), 1LL, multiply);

  // std::cout << "outer_dim: " << outer_dim << ", reduce_dim: " << reduce_dim
  // << ", inner_dim: " << inner_dim << std::endl;
  Ty *out_ptr = out_tensor.ptr();
  const Ty *in_ptr = input.ptr();
  for (auto outer = 0; outer < outer_dim; ++outer) {
    for (auto inner = 0; inner < inner_dim; ++inner) {
      auto base = inner + outer * reduce_dim * inner_dim;
      Ty max_val = std::numeric_limits<Ty>::lowest();

      // find max val
      for (auto reduce = 0; reduce < reduce_dim; ++reduce) {
        max_val = std::max(max_val, in_ptr[base + reduce * inner_dim]);
        out_ptr[base + reduce * inner_dim] = in_ptr[base + reduce * inner_dim];
      }

      // get exp sum
      Ty sum_val = static_cast<Ty>(0);
      for (auto reduce = 0; reduce < reduce_dim; ++reduce) {
        out_ptr[base + reduce * inner_dim] =
            std::exp(out_ptr[base + reduce * inner_dim] - max_val);
        sum_val += out_ptr[base + reduce * inner_dim];
      }

      // compute output
      for (auto reduce = 0; reduce < reduce_dim; ++reduce) {
        out_ptr[base + reduce * inner_dim] /= sum_val;
      }
    }
  }
  return std::move(out_tensor);
}
