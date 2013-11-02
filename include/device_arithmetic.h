#ifndef __DEVICE_BLAS_H_
#define __DEVICE_BLAS_H_

#include <device_matrix.h>
#ifndef __CUDACC__
#pragma message "\33[33mPotentially wrong compiler. Please use nvcc instead \33[0m"
#endif

// =====================================
// ===== Vector - Scalar Operators =====
// =====================================
#define VECTOR thrust::device_vector
#define WHERE thrust
#include <functional.inl>
#include <arithmetic.inl>
#undef VECTOR
#undef WHERE

#define dvec thrust::device_vector
#define dmat device_matrix

// ====================================
// ===== Vector Utility Functions =====
// ====================================
template <typename T>
T norm(const thrust::host_vector<T>& v) {
  return std::sqrt( thrust::transform_reduce(v.begin(), v.end(), func::square<T>(), 0, thrust::plus<T>()) );
}

template <typename T>
T norm(const thrust::device_vector<T>& v) {
  return std::sqrt( thrust::transform_reduce(v.begin(), v.end(), func::square<T>(), 0, thrust::plus<T>()) );
}

template <typename T>
void print(const thrust::host_vector<T>& v) {
  std::vector<T> stl_v(v.begin(), v.end());
  printf("[");
  for (size_t i=0; i<v.size(); ++i)
    printf("%.4f ", v[i]);
  printf("]\n");
}

template <typename T>
void print(const thrust::device_vector<T>& v) {
  thrust::host_vector<T> hv(v);
  print(hv);
}

// =====================================
// ===== Matrix - Vector Operators =====
// =====================================

template <typename T>
dmat<T> operator * (const dvec<T>& col_vector, const dvec<T>& row_vector) {
  size_t m = col_vector.size();
  size_t n = row_vector.size();
  dmat<T> result(m, n);
  size_t k = 1;

  // Treat device_vector as an 1 by N matrix
  const T* cv = thrust::raw_pointer_cast(col_vector.data());
  const T* rv = thrust::raw_pointer_cast(row_vector.data());

  float alpha = 1.0, beta = 0.0;

  int lda = m;
  int ldb = 1;
  int ldc = m;

  cublasStatus_t status;
  status = cublasSgemm(CUBLAS_HANDLE::getInstance(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, cv, lda, rv, ldb, &beta, result.getData(), ldc);

  CCE(status);

  return result;
}

template <typename T>
dvec<T> operator & (const dvec<T>& x, const dvec<T>& y) {
  assert(x.size() == y.size());
  dvec<T> z(x.size());
  thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), thrust::multiplies<T>());
  return z;
}

template <typename T>
dmat<T> operator * (const dvec<T>& v, const dmat<T>& A) {
  assert(v.size() == A.getRows());
  device_matrix<T> m(1, A.getCols());

  float alpha = 1.0, beta = 0.0;
  CCE(cublasSgemv(CUBLAS_HANDLE::getInstance(), CUBLAS_OP_T, A.getRows(), A.getCols(), &alpha, A.getData(), A.getRows(), thrust::raw_pointer_cast(v.data()), STRIDE, &beta, m.getData(), STRIDE));

  return m;
}

template <typename T>
dmat<T> operator * (const dmat<T>& A, const dvec<T>& v) {
  assert(A.getCols() == v.size());

  device_matrix<T> m(A.getRows(), 1);

  float alpha = 1.0, beta = 0.0;
  CCE(cublasSgemv(CUBLAS_HANDLE::getInstance(), CUBLAS_OP_N, A.getRows(), A.getCols(), &alpha, A.getData(), A.getRows(), thrust::raw_pointer_cast(v.data()), STRIDE, &beta, m.getData(), STRIDE));

  return m;
}

#undef dvec
#undef dmat

#endif // __DEVICE_BLAS_H_
