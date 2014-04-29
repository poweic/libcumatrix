#include <cuda_memory_manager.h>

template <typename T>
size_t CudaMemManager<T>::_cache_size = 65536;

template <typename T>
void CudaMemManager<T>::free(T* ptr) {
  // Un-comment the following line to see how much speed can gain from this.
  // CCE(cudaFree(ptr)); return;
  
  size_t byte = _byte_allocated[ptr];
  _byte_allocated.erase(ptr);
  this->push(byte, ptr);
  this->gc();
}

template <typename T>
T* CudaMemManager<T>::malloc(size_t N) {
  T* ptr;

  if (this->hasMore(N)) {
    // printf("\33[33m[Info]\33[0m Get %lu bytes memory from recycle pool\n", N);
    ptr = this->get(N);
  }
  else {
    // printf("\33[33m[Info]\33[0m Allocating %lu bytes memory.\n" , N);

    CCE(cudaMalloc((void **) &ptr, N * sizeof(T)));
    CCE(cudaDeviceSynchronize());

    _total_byte_allocated += N;
  }

  // Keep track of the memory size hold by each pointer.
  _byte_allocated[ptr] = N;

  return ptr;
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
