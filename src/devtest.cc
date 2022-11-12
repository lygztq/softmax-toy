#include "tensor.h"

int main() {
  CpuTensor<int> tensor_0 = CpuTensor<int>::cpu_tensor({4, 5});
  CudaTensor<int> device_tensor = tensor_0.to_cuda();
  CpuTensor<int> tensor = device_tensor.to_cpu();

  auto *dptr = tensor.ptr();
  for (int i = 0; i < 20; ++i) {
    dptr[i] = i;
  }

  CpuTensor<int> tensor_2 = tensor.copy();
  dptr = tensor_2.ptr();

  for (int i = 0; i < 20; ++i) {
    std::cout << dptr[i] << ", ";
  }

  return 0;
}