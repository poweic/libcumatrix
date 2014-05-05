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

// #include "boost/unordered_map.hpp"

template <typename T>
struct MemList {
  MemList(): hits(0), hit_rate(0) {}

  bool empty() const {
    return ptrs.empty();
  }

  size_t size() const {
    return ptrs.size();
  }

  T* back() const {
    return ptrs.back();
  }
  
  void pop_back() {
    ptrs.pop_back();
  }

  void push_back(T* ptr) {
    ptrs.push_back(ptr);
  }

  void hit() {
    ++hits;
    ++hit_rate;
  }

  std::vector<T*> ptrs;
  size_t hits;
  size_t hit_rate;
};


template <typename T>
struct MemPool {
  // typedef typename boost::unordered_map<size_t, MemList<T> > type;
  typedef typename std::map<size_t, MemList<T> > type;
};

template <typename T>
class CudaMemManager {
public:

  static T* malloc(size_t N);

  static void free(T* data);

  static void gc();
  
  static void showCacheHits();

  static void setCacheSize(size_t cache_size);

private:

  static CudaMemManager& getInstance() {
    static CudaMemManager memoryManager;
    return memoryManager;
  }

  CudaMemManager(): _total_byte_allocated(0) { /* Nothing to do */ }

  ~CudaMemManager() {}

  void push(size_t size, T* ptr);

  bool hasMore(size_t size);

  T* get(size_t size);

  void garbage_collection();

  size_t size() const;

  static size_t CACHE_SIZE;

  static std::vector<size_t> getKeys(const typename MemPool<T>::type &pool);
  static std::vector<size_t> sort_memlist_by_hits();
  static std::vector<size_t> sort_memlist_by_hit_rate();
  
  // Data member
  typename MemPool<T>::type _pool;
  size_t _total_byte_allocated;
  std::map<T*, size_t> _byte_allocated;
};

#endif // __CUDA_MEMORY_MANAGER_H_
