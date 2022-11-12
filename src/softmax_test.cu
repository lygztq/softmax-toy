#include "softmax_cpu.h"
#include "softmax_cuda.cuh"
#include "tensor.h"
#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>

int main() {
  CpuTensor<float> host_tensor = CpuTensor<float>::cpu_tensor({512, 2048, 16});

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 2.0);

  float *ptr = host_tensor.ptr();
  for (auto i = 0; i < host_tensor.nelem(); ++i) {
    ptr[i] = dis(gen);
  }

  CudaTensor<float> device_tensor = host_tensor.to_cuda();

  float cuda_time = 0;
  cudaEvent_t cuda_start, cuda_end;
  cudaEventCreate(&cuda_start);
  cudaEventCreate(&cuda_end);
  
  cudaEventRecord(cuda_start);
  CudaTensor<float> tmp_out = softmax_cuda<float>(device_tensor, 1);
  cudaEventRecord(cuda_end);
  cudaEventSynchronize(cuda_end);
  cudaEventElapsedTime(&cuda_time, cuda_start, cuda_end);
  
  CpuTensor<float> device_out = tmp_out.to_cpu();

  auto cpu_start = std::chrono::steady_clock::now();
  CpuTensor<float> host_out = softmax_cpu<float>(host_tensor, 1);
  auto cpu_end = std::chrono::steady_clock::now();
  std::chrono::duration<float, std::milli> cpu_time = cpu_end - cpu_start;

  const float *host_out_ptr = host_out.ptr();
  const float *device_out_ptr = device_out.ptr();
  for (auto i = 0; i < host_out.nelem(); ++i) {
    auto diff = std::abs(host_out_ptr[i] - device_out_ptr[i]);
    if (diff > 1e-5) {
      std::cerr << "Find mismatch at global idx " << i
                << ", host out: " << host_out_ptr[i]
                << ", device out: " << device_out_ptr[i] << ", diff: " << diff
                << std::endl;
      return 1;
    }
  }

  std::cout << "All passed!, " << std::endl
            << "CPU time: " << cpu_time.count() << "ms" << std::endl
            << "CUDA time: " << cuda_time << "ms" << std::endl;
  return 0;
}