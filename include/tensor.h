#ifndef TENSOR_H_
#define TENSOR_H_

#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#define LOG_ERROR(msg)                                                         \
  {                                                                            \
    std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ": " << msg       \
              << std::endl;                                                    \
    throw std::runtime_error(msg);                                             \
  }

enum class DeviceTag {
  kCpu,
  kCuda,
  kInvalid,
};

struct TensorBase {
  std::vector<int64_t> shape;
  DeviceTag device_tag;

  TensorBase() : device_tag(DeviceTag::kInvalid) {}
  explicit TensorBase(const std::vector<int64_t> &shape, DeviceTag tag)
      : shape(shape), device_tag(tag) {}

  TensorBase(const TensorBase &) = delete;
  TensorBase &operator=(const TensorBase &) = delete;
  TensorBase(TensorBase &&other)
      : shape(std::move(other.shape)), device_tag(other.device_tag) {}

  friend void swap(TensorBase &t0, TensorBase &t1) {
    using std::swap;
    swap(t0.shape, t1.shape);
    swap(t0.device_tag, t1.device_tag);
  }

  TensorBase &operator=(TensorBase &&other) {
    using std::swap;
    auto tmp = std::move(other);
    swap(tmp, *this);
    return *this;
  }

  virtual ~TensorBase() {}

  static int64_t compute_nelem(const std::vector<int64_t> &shape) {
    return std::reduce(shape.begin(), shape.end(), 1LL,
                       [](int64_t s0, int64_t s1) { return s0 * s1; });
  }

  int64_t ndim() const { return static_cast<int64_t>(shape.size()); }
  int64_t nelem() const { return compute_nelem(this->shape); }
  int64_t nbyte(int64_t dtype_size) const { return dtype_size * this->nelem(); }
  bool is_cuda() const { return device_tag == DeviceTag::kCuda; }
  bool is_cpu() const { return device_tag == DeviceTag::kCpu; }
};

template <typename Ty> struct CpuTensor;

template <typename Ty> struct CudaTensor : public TensorBase {
  static void default_delete(Ty *ptr) {
    if (ptr)
      cudaFree(ptr);
  }
  CudaTensor() : TensorBase() {}
  explicit CudaTensor(Ty *data, const std::vector<int64_t> &shape)
      : TensorBase(shape, DeviceTag::kCuda), data(data, &default_delete) {}
  ~CudaTensor() { this->data.reset(); }

  CudaTensor(CudaTensor &&other)
      : TensorBase(std::move(other)), data(std::move(other.data)) {}
  CudaTensor &operator=(CudaTensor &&other) {
    using std::swap;
    auto tmp = std::move(other);
    swap(tmp, *this);
    return *this;
  }

  friend void swap(CudaTensor<Ty> &t1, CudaTensor<Ty> &t2) {
    using std::swap;
    swap(t1.data, t2.data);
    swap(static_cast<TensorBase &>(t2), static_cast<TensorBase &>(t2));
  }

  Ty *ptr() { return data.get(); }
  const Ty *ptr() const { return data.get(); }

  static CudaTensor<Ty> cuda_tensor(const std::vector<int64_t> &shape) {
    auto num_bytes = TensorBase::compute_nelem(shape) * sizeof(Ty);
    Ty *device_ptr;
    auto state = cudaMalloc(&device_ptr, num_bytes);

    if (state != cudaSuccess) {
      LOG_ERROR(cudaGetErrorString(state));
    }
    return CudaTensor<Ty>(device_ptr, shape);
  }

  static constexpr int64_t dtype_size = sizeof(Ty);

  CpuTensor<Ty> to_cpu() const {
    CpuTensor<Ty> cpu_tensor = CpuTensor<Ty>::cpu_tensor(this->shape);
    const auto *device_ptr = this->ptr();
    auto *host_ptr = cpu_tensor.ptr();
    auto state = cudaMemcpy(host_ptr, device_ptr, this->nbyte(dtype_size),
                            cudaMemcpyDeviceToHost);
    if (state != cudaSuccess) {
      LOG_ERROR("Error when copy tensor from device to host");
    }
    return std::move(cpu_tensor);
  }

  CudaTensor<Ty> copy() const {
    CudaTensor<Ty> cuda_tensor = CudaTensor<Ty>::cuda_tensor(this->shape);
    auto state = cudaMemcpy(cuda_tensor.ptr(), this->ptr(),
                            this->nbyte(dtype_size), cudaMemcpyDeviceToDevice);
    if (state != cudaSuccess) {
      LOG_ERROR("Error when copy tensor from device to device");
    }
    return std::move(cuda_tensor);
  }

private:
  using data_ptr_t =
      std::unique_ptr<Ty, decltype(&CudaTensor<Ty>::default_delete)>;
  data_ptr_t data = nullptr;
};

template <typename Ty> struct CpuTensor : public TensorBase {
  CpuTensor() : TensorBase() {}
  explicit CpuTensor(Ty *data, const std::vector<int64_t> &shape)
      : TensorBase(shape, DeviceTag::kCpu), data(data) {}
  ~CpuTensor() { data.reset(); }

  CpuTensor(CpuTensor &&other)
      : TensorBase(std::move(other)), data(std::move(other.data)) {}

  CpuTensor &operator=(CpuTensor &&other) {
    using std::swap;
    auto tmp = std::move(other);
    swap(tmp, *this);
    return *this;
  }

  friend void swap(CpuTensor<Ty> &t1, CpuTensor<Ty> &t2) {
    using std::swap;
    swap(t1.data, t2.data);
    swap(static_cast<TensorBase &>(t2), static_cast<TensorBase &>(t2));
  }

  Ty *ptr() { return data.get(); }
  const Ty *ptr() const { return data.get(); }

  static CpuTensor<Ty> cpu_tensor(const std::vector<int64_t> &shape) {
    auto num_elems = TensorBase::compute_nelem(shape);
    Ty *cpu_ptr = new Ty[num_elems];

    if (!cpu_ptr) {
      LOG_ERROR("error when alloc cpu data ptr.");
    }

    CpuTensor<Ty> tensor(cpu_ptr, shape);
    return std::move(tensor);
  }

  static constexpr int64_t dtype_size = sizeof(Ty);

  CudaTensor<Ty> to_cuda() const {
    CudaTensor<Ty> cuda_tensor = CudaTensor<Ty>::cuda_tensor(this->shape);
    const auto *host_ptr = this->ptr();
    auto *device_ptr = cuda_tensor.ptr();
    auto state = cudaMemcpy(device_ptr, host_ptr, this->nbyte(dtype_size),
                            cudaMemcpyHostToDevice);
    if (state != cudaSuccess) {
      LOG_ERROR("Error when copy tensor from device to host");
    }
    return std::move(cuda_tensor);
  }

  CpuTensor<Ty> copy() const {
    CpuTensor<Ty> cpu_tensor = CpuTensor<Ty>::cpu_tensor(this->shape);
    std::copy_n(this->ptr(), this->nelem(), cpu_tensor.ptr());
    return std::move(cpu_tensor);
  }

private:
  std::unique_ptr<Ty[]> data;
};

#endif // TENSOR_H_