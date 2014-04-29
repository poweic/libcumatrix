#include <cuda_memory_manager.h>

template <typename T>
CudaMemManager<T>::~CudaMemManager() {

  /*while (!_data_to_free.empty()) {
    T* d = _data_to_free.front();

    cudaError_t e = cudaFree(d);
    if (e != cudaErrorCudartUnloading)
      checkCudaErrors(e);

    _data_to_free.pop();
  }*/
}

template <typename T>
void CudaMemManager<T>::free(T* data) {
  static size_t queued_bytes = 0;
  // Un-comment the following line to see how much speed can gain from this.
  // CCE(cudaFree(data)); return;
  _data_to_free.push(data);

  size_t byte = _byte_allocated[data];
  queued_bytes += byte;

  if (queued_bytes >= MEM_BLOCK) {
    this->free_all();
    queued_bytes = 0;
  }
}

template <typename T>
T* CudaMemManager<T>::malloc(size_t N) {
  T* data;
  CCE(cudaMalloc((void **) &data, N * sizeof(T)));
  _byte_allocated[data] = N;
  _total_byte_allocated += N;

  CCE(cudaDeviceSynchronize());
  return data;
}

template <typename T>
void CudaMemManager<T>::free_all() {
  while (!_data_to_free.empty()) {
    T* d = _data_to_free.front();

    if (d) {
      CCE(cudaFree(d));
      _byte_allocated.erase(d);
    }

    _data_to_free.pop();
  }

  CCE(cudaDeviceSynchronize());
}

template class CudaMemManager<float>;
template class CudaMemManager<double>;
