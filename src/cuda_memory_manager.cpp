#include <cuda_memory_manager.h>
#include <algorithm>

template <typename T>
size_t CudaMemManager<T>::CACHE_SIZE = 16 * 1024 * 1024; /* 16 MBytes */

template <typename T>
T* CudaMemManager<T>::malloc(size_t N) {

  CudaMemManager& memMgr = getInstance();
  T* ptr;

  if (memMgr.hasMore(N)) {
    // printf("\33[33m[Info]\33[0m Get %lu bytes memory from recycle pool\n", N);
    ptr = memMgr.get(N);
  }
  else {
    // printf("\33[33m[Info]\33[0m Allocating %lu bytes memory.\n" , N);

    CCE(cudaMalloc((void **) &ptr, N * sizeof(T)));
    CCE(cudaDeviceSynchronize());

    // printf("_total_byte_allocated: %lu => %lu\n", memMgr._total_byte_allocated, memMgr._total_byte_allocated + N);
    memMgr._total_byte_allocated += N;
  }

  // Keep track of the memory size hold by each pointer.
  memMgr._byte_allocated[ptr] = N;

  return ptr;
}

template <typename T>
void CudaMemManager<T>::free(T* ptr) {
  // Un-comment the following line to see how much speed can gain from this.
  // CCE(cudaFree(ptr)); return;
  
  CudaMemManager& memMgr = getInstance();

  size_t byte = memMgr._byte_allocated[ptr];
  memMgr._byte_allocated.erase(ptr);
  memMgr.push(byte, ptr);
  memMgr.gc();
}

template <typename T>
void CudaMemManager<T>::gc() {
  getInstance().garbage_collection();
}


template <typename T>
void CudaMemManager<T>::push(size_t size, T* ptr) {

  if (size == 0)
    return;

  if (_pool.count(size) == 0)
    _pool[size] = MemList<T>();

  _pool[size].push_back(ptr);
}

template <typename T>
bool CudaMemManager<T>::hasMore(size_t size) {
  return _pool.count(size) > 0 && !_pool[size].empty();
}

template <typename T>
T* CudaMemManager<T>::get(size_t size) {
  _pool[size].hit();
  T* ptr = _pool[size].back();
  _pool[size].pop_back();
  return ptr;
}

template <typename T>
size_t CudaMemManager<T>::size() const {
  return getInstance()._total_byte_allocated;
}

template <typename T>
void CudaMemManager<T>::setCacheSize(size_t cache_size_in_MB) {
  CACHE_SIZE = cache_size_in_MB * 1024 * 1024;
}

template <typename T>
std::vector<size_t> CudaMemManager<T>::getKeys(const typename MemPool<T>::type &pool) {
  std::vector<size_t> keys;
  keys.reserve(pool.size());

  typename MemPool<T>::type::const_iterator itr;
  for (itr = pool.begin(); itr != pool.end(); ++itr)
    keys.push_back(itr->first);

  return keys;
}

template <typename T>
std::vector<size_t> CudaMemManager<T>::sort_memlist_by_hits() {
  CudaMemManager& memMgr = getInstance();

  std::vector<size_t> keys = getKeys(memMgr._pool);

  std::sort(keys.begin(), keys.end(), [&memMgr](size_t a, size_t b) {
    return memMgr._pool[a].hits < memMgr._pool[b].hits;
  });

  return keys;
}

template <typename T>
std::vector<size_t> CudaMemManager<T>::sort_memlist_by_hit_rate() {
  CudaMemManager& memMgr = getInstance();

  std::vector<size_t> keys = getKeys(memMgr._pool);

  std::sort(keys.begin(), keys.end(), [&memMgr](size_t a, size_t b) {
    return memMgr._pool[a].hit_rate < memMgr._pool[b].hit_rate;
  });

  return keys;
}


template <typename T>
void CudaMemManager<T>::garbage_collection() {
  
  CudaMemManager& memMgr = getInstance();

  size_t& bytes = memMgr._total_byte_allocated;
  if (bytes <= CACHE_SIZE)
    return;
  
  auto keys = sort_memlist_by_hit_rate();
  auto& p = memMgr._pool;

  size_t before_free = bytes;
  
  // printf("keys.size() = %lu\n", keys.size());
  for (size_t i=0; i<keys.size(); ++i) {
    size_t k = keys[i];
    // printf("keys[%lu] = %lu\n", i, keys[i]);

    while ( bytes > CACHE_SIZE && !p[k].empty() ) {
      // printf("pool[%lu].size() = %lu\n", k, p[k].size());
      T* ptr = p[k].back();
      CCE(cudaFree(ptr));
      p[k].pop_back();
      bytes -= k;
      memMgr._byte_allocated.erase(ptr);
    }

    // Reset hit rate
    p[k].hit_rate = 0;
  }

  CCE(cudaDeviceSynchronize());
  // printf("\33[33m[Info]\33[0m Byte allocated: %lu => %lu\n", before_free, bytes);
}

template <typename T>
void CudaMemManager<T>::showCacheHits() {
  CudaMemManager& memMgr = getInstance();
  typename MemPool<T>::type::const_iterator itr;

  std::vector<size_t> keys = sort_memlist_by_hits();
  size_t maxHits = memMgr._pool[keys.back()].hits;

  const size_t BAR_LENGTH = 60;

  printf("\n");
  printf("+----------+------------------------------------------------------------+\n");
  printf("| Mem Size |                        Memory Usage                        |\n");
  printf("|   (KB)   |                        ( # of hit )                        |\n");
  printf("+----------+------------------------------------------------------------+\n");

  for (size_t i=0; i<keys.size(); ++i) {
    printf("|%9.2f |", (float) keys[i] / 1024);

    size_t length = (size_t) ((double) (memMgr._pool[keys[i]].hits * BAR_LENGTH) / maxHits);
    if (length == 0)
      length = 1;

    printf("\33[32m");
    for (size_t i=0; i<BAR_LENGTH; ++i) {
      if (i == 30) printf("\33[33m");
      if (i == 53) printf("\33[31m");

      if (i < length)
	printf("|");
      else
	printf(" ");
    }
    printf("\33[0m|\n");
  }
  printf("+----------+------------------------------------------------------------+\n");
}

template class CudaMemManager<float>;
template class CudaMemManager<double>;
