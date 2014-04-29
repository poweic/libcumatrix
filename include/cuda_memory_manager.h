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

#include "boost/unordered_map.hpp"

template <typename T>
struct MemPool {
  typedef typename boost::unordered_map<size_t, std::vector<T*> > type;
  // typedef typename std::map<size_t, std::vector<T*> > type;
};

template <typename T>
class CudaMemManager {
public:
  CudaMemManager(): _total_byte_allocated(0) { /* Nothing to do */ }
  ~CudaMemManager() {}

  void free(T* data);
  T* malloc(size_t N);

private:

  void push(size_t size, T* ptr) {

    if (size == 0)
      return;

    if (_pool.count(size) == 0)
      _pool[size] = std::vector<T*>();

    std::vector<T*> &x = _pool[size];
    
    x.push_back(ptr);
  }

  bool hasMore(size_t size) {
    return _pool.count(size) > 0 && !_pool[size].empty();
  }

  T* get(size_t size) {
    assert(_pool[size].size() > 0);

    T* ptr = _pool[size].back();
    _pool[size].pop_back();
    return ptr;
  }

  void gc() { this->garbage_collection(); }

  void garbage_collection() {
    // printf("\33[33m[Info]\33[0m Total %lu bytes allocated.\n" , _total_byte_allocated);

    // TODO do something to free
    //
    // TODO count how much freed, update _total_byte_allocated
    // 
    // TODO update _byte_allocated
  }

  size_t size() const {
    return _total_byte_allocated;
  }

  void free_all();

  static void setCacheSize(size_t cache_size) {
    _cache_size = cache_size;
  }

  static size_t _cache_size;
  
  typename MemPool<T>::type _pool;

  size_t _total_byte_allocated;
  std::map<T*, size_t> _byte_allocated;
  std::queue<T*> _data_to_free;

};

#endif // __CUDA_MEMORY_MANAGER_H_
