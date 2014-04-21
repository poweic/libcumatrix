#ifndef __CUDA_MEMORY_MANAGER_H_
#define __CUDA_MEMORY_MANAGER_H_

#include <iostream>
using std::cout;
using std::endl;

#include <cuda_runtime.h>
#include <helper_cuda.h>
#define CCE(x) checkCudaErrors(x)

#include <queue>
#include <map>

template <typename T>
class CudaMemManager {
public:
  CudaMemManager(): _total_byte_allocated(0) { /* Nothing to do */ }
  ~CudaMemManager();

  void free(T* data);
  T* malloc(size_t N);

private:

  void free_all();

  static void free_all(std::queue<std::pair<T*, size_t> > data, size_t& nPopped);
  static const size_t MEM_BLOCK = 65536;

  size_t _total_byte_allocated;
  std::map<T*, size_t> _byte_allocated;
  std::queue<T*> _data_to_free;
};

#endif // __CUDA_MEMORY_MANAGER_H_
