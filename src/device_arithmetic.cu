#include <device_arithmetic.h>

#define dmat device_matrix
#define dvec thrust::device_vector

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

template <typename T>
dmat<T> operator + (T val, const dmat<T>& m) {
  return m + val;
}

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

template <typename T>
dmat<T> operator & (const dvec<T>& v, const dmat<T>& m) {

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // FIXME !!!!!!!!!!!!!! THIS IS FUCKING SLOW !!!!!!!!!!!!!!!!! FIXME
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  assert(v.size() == m.getCols());
  dmat<T> result(m);

  size_t rows = m.getRows();
  size_t cols = m.getCols();

  for (size_t i=0; i<cols; ++i) {
    thrust::device_ptr<T> ptr(m.getData() + rows * i);
    thrust::device_ptr<T> ptr2(result.getData() + rows * i);

    dvec<T> cv(ptr, ptr + rows);
    dvec<T> cv2(ptr2, ptr2 + rows);

    thrust::transform(
	m.getData() + rows * i,
	m.getData() + rows * (i + 1),
	result.getData() + rows * i,
	func::ax<T>(v[i]) );
  }

  return result;
}

#define EXPLICITLY_INSTANTIATE(T) \
template T norm<T> (const thrust::host_vector<T>& v); \
template T norm<T> (const thrust::device_vector<T>& v); \
template void print<T> (const thrust::host_vector<T>& v); \
template void print<T> (const thrust::device_vector<T>& v); \
template dmat<T> operator & <T> (const dvec<T>& v, const dmat<T>& m); \
template dmat<T> operator * <T> (const dvec<T>& col_vector, const dvec<T>& row_vector); \
template dvec<T> operator & <T> (const dvec<T>& x, const dvec<T>& y); \
template dmat<T> operator * <T> (const dvec<T>& v, const dmat<T>& m); \
template dmat<T> operator * <T> (const dmat<T>& m, const dvec<T>& v);

EXPLICITLY_INSTANTIATE(float);

#undef dmat
#undef dvec
