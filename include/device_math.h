#ifndef __DEVICE_MATH_EXT_H_
#define __DEVICE_MATH_EXT_H_

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#define HAVE_THRUST_DEVICE_VECTOR_H 1

#include <device_matrix.h>

namespace ext {
  template <typename T>
  vector<T> toStlVector(const thrust::device_vector<T>& v) {
    vector<T> stl_vector(v.size());
    thrust::copy(v.begin(), v.end(), stl_vector.begin());
    return stl_vector;
  }

  // ========================
  // ===== Save as File =====
  // ========================
  template <typename T>
  void save(const vector<T>& v, string filename) {
    ofstream fs(filename.c_str());

    fs.precision(6);
    fs << std::scientific;
    for (size_t i=0; i<v.size(); ++i)
      fs << v[i] << endl;

    fs.close();
  }

  template <typename T>
  void save(const thrust::device_vector<T>& v, string filename) {
    ext::save(toStlVector(v), filename);
  }

  // ==========================
  // ===== Load from File =====
  // ==========================
  template <typename T>
  void load(vector<T>& v, string filename) {
    v.clear();

    ifstream fs(filename.c_str());

    T t;
    while (fs >> t) 
      v.push_back(t);

    fs.close();
  }

  template <typename T>
  thrust::device_vector<T> load(string filename) {
    vector<T> hv;
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
