#ifndef __DEVICE_MATH_EXT_H_
#define __DEVICE_MATH_EXT_H_

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#define HAVE_THRUST_DEVICE_VECTOR_H 1

#include <math_ext.h>
#include <functional.inl>
#include <device_matrix.h>

namespace ext {
  template <typename T>
  std::vector<T> toStlVector(const thrust::device_vector<T>& v) {
    std::vector<T> stl_vector(v.size());
    thrust::copy(v.begin(), v.end(), stl_vector.begin());
    return stl_vector;
  }

  // ========================
  // ===== Save as File =====
  // ========================

  template <typename T>
  void save(const thrust::device_vector<T>& v, std::string filename) {
    ext::save(toStlVector(v), filename);
  }

  // ==========================
  // ===== Load from File =====
  // ==========================

  template <typename T>
  thrust::device_vector<T> load(std::string filename) {
    std::vector<T> hv;
    ext::load<T>(hv, filename);
    return thrust::device_vector<T>(hv.begin(), hv.end());
  }

  // =================
  // ===== Print =====
  // =================
  template <typename T>
  void print(const thrust::host_vector<T>& v) {
    std::vector<T> stl_v(v.begin(), v.end());
    printf("[");
    for (size_t i=0; i<v.size(); ++i)
      printf("%.4f ", v[i]);
    printf("]\n\n");
  }

  template <typename T>
  void print(const thrust::device_vector<T>& v) {
    thrust::host_vector<T> hv(v);
    print(hv);
  }

  template <typename T>
  void rand(device_matrix<T>& m) {
    T* data = new T[m.size()];

    for (size_t i=0; i<m.size(); ++i)
      data[i] = rand01<T>();

    CCE(cudaMemcpy(m.getData(), data, sizeof(T) * m.size(), cudaMemcpyHostToDevice));
    delete [] data;
  }


  template <typename T>
  void randn(device_matrix<T>& m, float mean = 0.0, float variance = 1.0) {
    T* data = new T[m.size()];

    for (size_t i=0; i<m.size(); ++i)
      data[i] = randn<T>(mean, variance);

    CCE(cudaMemcpy(m.getData(), data, sizeof(T) * m.size(), cudaMemcpyHostToDevice));
    delete [] data;
  }

  // =====================
  // ===== L2 - norm =====
  // =====================
  template <typename T>
  T norm(const thrust::host_vector<T>& v) {
    return std::sqrt( thrust::transform_reduce(v.begin(), v.end(), func::square<T>(), (T) 0, thrust::plus<T>()) );
  }

  template <typename T>
  T norm(const thrust::device_vector<T>& v) {
    return std::sqrt( thrust::transform_reduce(v.begin(), v.end(), func::square<T>(), (T) 0, thrust::plus<T>()) );
  }

  // ===============
  // ===== Sum =====
  // ===============
  template <typename T>
  T sum(const thrust::device_vector<T>& v) {
    return thrust::reduce(v.begin(), v.end());
  }

  template <typename T>
  T sum(const device_matrix<T>& m) {
    return thrust::reduce(m.getData(), m.getData() + m.size(), (T) 0, thrust::plus<T>());
  }

};

#endif // __DEVICE_MATH_EXT_H_
