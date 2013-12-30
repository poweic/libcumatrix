#include <device_matrix.h>
#define mylog(token) {cout << #token " = " << token << endl;}

// ===============================
// ===== class device_matrix =====
// ===============================

template <typename T>
device_matrix<T>::device_matrix():
  _transposed(false),
  _rows(0), _cols(0),
  _capacity(_rows * _cols),
  _data(NULL) { }

template <typename T>
device_matrix<T>::device_matrix(size_t r, size_t c):
  _transposed(false),
  _rows(r), _cols(c),
  _capacity(_rows*_cols),
  _data(NULL) {

  _init();
  fillwith(0);
}

template <typename T>
device_matrix<T>::device_matrix(T* h_data, size_t r, size_t c):
  _transposed(false),
  _rows(r), _cols(c),
  _capacity(_rows*_cols),
  _data(NULL) {

  _init();
  CCE(cudaMemcpy(_data, h_data, sizeof(T) * _rows * _cols, cudaMemcpyHostToDevice));
}

template <typename T>
device_matrix<T>::device_matrix(const string& filename):
  _transposed(false),
  _rows(0), _cols(0),
  _capacity(_rows*_cols),
  _data(NULL) {

  const size_t MAX_BUFFER = 262144;
  char line[MAX_BUFFER];

  FILE* fid = fopen(filename.c_str(), "r");
  while (fgets(line, MAX_BUFFER, fid)) {
    _rows++;

    assert(line[strlen(line) - 1] == '\n');

    if (_cols != 0)
      continue;

    char* token = strtok(line, " \n");
    ++_cols;
    while(strtok(NULL, " \n"))
      ++_cols;
  }
  fseek(fid, 0, SEEK_SET);

  // BEWARE !!
  // BLAS stores data in column-major
  const char *rspecifier = (sizeof(T) / sizeof(float) == 1) ? "%f" : "%lf";

  T* data = new T[_rows*_cols];
  for (size_t i=0; i<_rows; ++i)
    for (size_t j=0; j<_cols; ++j)
      fscanf(fid, rspecifier, &(data[j*_rows + i]));
  fclose(fid);

  _init();

  CCE(cudaMemcpy(_data, data, sizeof(T) * _rows * _cols, cudaMemcpyHostToDevice));
  delete [] data;
}
// Copy Constructor 
template <typename T>
device_matrix<T>::device_matrix(const device_matrix<T>& source):
  _transposed(false),
  _rows(source._rows), _cols(source._cols),
  _capacity(_rows * _cols),
  _data(NULL) {

  _init();
  if (!source._transposed)
    CCE(cudaMemcpy(_data, source._data, sizeof(T) * _rows * _cols, cudaMemcpyDeviceToDevice));
  else {
    // TODO
    // Transpose and copy at the same time (see Nvidia CUDA forum)
  }
}

#ifdef HAVE_THRUST_DEVICE_VECTOR_H
// Conversion operator
template <typename T>
device_matrix<T>::operator thrust::device_vector<T>() const {
  assert(_rows == 1 || _cols == 1);
  return thrust::device_vector<T>(_data, _data + size());
}
#endif

template <typename T>
device_matrix<T>::~device_matrix() {
  CCE(cudaFree(_data));
}

// ===========================
// ===== Other Functions =====
// ===========================

// ===== Addition =====
template <typename T>
device_matrix<T>& device_matrix<T>::operator += (T val) {
  cublas_axpy(_rows*_cols, val, SCALAR_MEMORY_BUFFER<T>::getBuffer(), 0, _data, 1);
  return *this;
} 

template <typename T>
device_matrix<T> device_matrix<T>::operator + (T val) const {
  device_matrix<T> m(*this);
  return (m += val);
}

template <typename T>
device_matrix<T>& device_matrix<T>::operator += (device_matrix<T>& rhs) {
  *this = *this + rhs;
  return *this;
}

template <typename T>
device_matrix<T> device_matrix<T>::operator + (device_matrix<T>& rhs) {
  device_matrix<T> result(_rows, _cols);
  geam(*this, rhs, result, (T) 1.0, (T) 1.0);

  _transposed = rhs._transposed = false;
  return result;
}

// ===== Substraction =====
template <typename T>
device_matrix<T>& device_matrix<T>::operator -= (T val) {
  val = -val;
  cublas_axpy(_rows*_cols, val, SCALAR_MEMORY_BUFFER<T>::getBuffer(), 0, _data, 1);
  return *this;
}

template <typename T>
device_matrix<T> device_matrix<T>::operator - (T val) const {
  device_matrix<T> m(*this);
  return (m -= val);
}

template <typename T>
device_matrix<T>& device_matrix<T>::operator -= (device_matrix<T>& rhs) {
  *this = *this - rhs;
  return *this;
}

template <typename T>
device_matrix<T> device_matrix<T>::operator - (device_matrix<T>& rhs) {
  device_matrix<T> result(_rows, _cols);
  geam(*this, rhs, result, (T) 1.0, (T) -1.0);

  _transposed = rhs._transposed = false;
  return result;
}

// ===== Division =====
template <typename T>
device_matrix<T>& device_matrix<T>::operator /= (T alpha) {
  return *this *= ( (T) 1 / alpha );
}

template <typename T>
device_matrix<T> device_matrix<T>::operator / (T alpha) const {
  return *this * ( (T) 1 / alpha );
}

// ===== Matrix-scalar Multiplication =====
template <typename T>
device_matrix<T>& device_matrix<T>::operator *= (T alpha) {
  cublas_scal(_rows*_cols, alpha, _data, 1);
  return *this;
}

template <typename T>
device_matrix<T> device_matrix<T>::operator * (T alpha) const {
  device_matrix<T> result(*this);
  return result *= alpha;
}

// ===== Matrix-Matrix Multiplication =====
template <typename T>
device_matrix<T>& device_matrix<T>::operator *= (device_matrix<T>& rhs) {
  *this = *this * rhs;
  return *this;
}

template <typename T>
device_matrix<T> device_matrix<T>::operator * (device_matrix<T>& rhs) {
  device_matrix<T> result(_rows, rhs._cols);
  gemm(*this, rhs, result);
  
  _transposed = rhs._transposed = false;
  return result;
}

// Operator Assignment:
// call copy constructor first, and swap with the temp variable
template <typename T>
device_matrix<T>& device_matrix<T>::operator = (device_matrix<T> rhs) {
  swap(*this, rhs);
  return *this;
}

// Operator transpose
template <typename T>
device_matrix<T>& device_matrix<T>::operator ~ () {
  this->_transposed = true;
  return *this;
}

template <typename T>
void device_matrix<T>::_init() {
  _capacity = _rows * _cols;
  CCE(cudaMalloc((void **)&_data, _rows * _cols * sizeof(T)));
}

template <typename T>
void device_matrix<T>::resize(size_t r, size_t c) {
  // printf("trying to resize from (%lu, %lu) => (%lu, %lu), with original capacity = %lu\n", _rows, _cols, r, c, _capacity);
  if (_rows == r && _cols == c)
    return;

  _rows = r;
  _cols = c;

  if (r * c <= _capacity)
    return;

  CCE(cudaFree(_data));
  _init();
  fillwith(0);
}

template <typename T>
void device_matrix<T>::reserve(size_t capacity) {
  if (capacity <= _capacity)
    return;

  _capacity = capacity;

  T* buffer;
  CCE(cudaMalloc((void **)&buffer, _capacity * sizeof(T)));
  CCE(cudaMemset(buffer, 0, _capacity * sizeof(T)));
  CCE(cudaMemcpy(buffer, _data, sizeof(T) * size(), cudaMemcpyDeviceToDevice));
  CCE(cudaFree(_data));
  _data = buffer;
}

template <typename T>
void device_matrix<T>::print(FILE* fid) const {

  T* data = new T[size()];
  CCE(cudaMemcpy(data, _data, sizeof(T) * size(), cudaMemcpyDeviceToHost));

  for (size_t i=0; i<_rows; ++i) {
    for (size_t j=0; j<_cols; ++j)
      fprintf(fid, "%g ", data[j*_rows + i]);
    fprintf(fid, "\n");
  }

  if (fid == stdout)
    fprintf(fid, "\n");

  delete [] data;
}

template <typename T>
void device_matrix<T>::fillwith(T val) {
  cudaMemset(_data, 0, _rows * _cols * sizeof(T));

  if (val != 0)
    *this += val;
}

template <typename T>
void device_matrix<T>::save(const string& filename) const {
  FILE* fid = fopen(filename.c_str(), "w");
  if (fid == NULL)
    return;

  print(fid);
  fclose(fid);
}

template <>
void device_matrix<float>::cublas_gemm(
  cublasOperation_t transA, cublasOperation_t transB,
  int m, int n, int k,
  float alpha,
  const float* A, int lda,
  const float* B, int ldb,
  float beta,
  float* C, int ldc) {
  CCE(cublasSgemm(CUBLAS_HANDLE::getInstance(), transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

template <>
void device_matrix<double>::cublas_gemm(
  cublasOperation_t transA, cublasOperation_t transB,
  int m, int n, int k,
  double alpha,
  const double* A, int lda,
  const double* B, int ldb,
  double beta,
  double* C, int ldc) {
  CCE(cublasDgemm(CUBLAS_HANDLE::getInstance(), transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}

template <>
void device_matrix<float>::cublas_geam(
    cublasOperation_t transA, cublasOperation_t transB,
    int m, int n,
    float alpha, const float *A, int lda,
    float beta , const float *B, int ldb,
    float *C, int ldc) {
  CCE(cublasSgeam(CUBLAS_HANDLE::getInstance(), transA, transB, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc));
}

template <>
void device_matrix<double>::cublas_geam(
    cublasOperation_t transA, cublasOperation_t transB,
    int m, int n,
    double alpha, const double *A, int lda,
    double beta , const double *B, int ldb,
    double *C, int ldc) {
  CCE(cublasDgeam(CUBLAS_HANDLE::getInstance(), transA, transB, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc));
}

template <>
void device_matrix<float>::cublas_gemv(
    cublasOperation_t trans,
    int m, int n,
    float alpha,
    const float *A, int lda,
    const float *x, int incx,
    float beta,
    float *y, int incy) {
  CCE(cublasSgemv(CUBLAS_HANDLE::getInstance(), trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy));
}

template <>
void device_matrix<double>::cublas_gemv(
    cublasOperation_t trans,
    int m, int n,
    double alpha,
    const double *A, int lda,
    const double *x, int incx,
    double beta,
    double *y, int incy) {
  CCE(cublasDgemv(CUBLAS_HANDLE::getInstance(), trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy));
}

template <>
void device_matrix<float>::cublas_iamax(int n, const float *x, int incx, int *result) {
  CCE(cublasIsamax(CUBLAS_HANDLE::getInstance(), n, x, incx, result));
}

template <>
void device_matrix<double>::cublas_iamax(int n, const double *x, int incx, int *result) {
  CCE(cublasIdamax(CUBLAS_HANDLE::getInstance(), n, x, incx, result));
}


template <>
void device_matrix<float>::cublas_nrm2(int n, const float *x, int incx, float *result) {
  CCE(cublasSnrm2(CUBLAS_HANDLE::getInstance(), n, x, incx, result));
}

template <>
void device_matrix<double>::cublas_nrm2(int n, const double *x, int incx, double *result) {
  CCE(cublasDnrm2(CUBLAS_HANDLE::getInstance(), n, x, incx, result));
}

template <>
void device_matrix<float>::cublas_scal(int n, float alpha, float *x, int incx) {
  CCE(cublasSscal(CUBLAS_HANDLE::getInstance(), n, &alpha, x, incx));
}

template <>
void device_matrix<double>::cublas_scal(int n, double alpha, double *x, int incx) {
  CCE(cublasDscal(CUBLAS_HANDLE::getInstance(), n, &alpha, x, incx));
}

template <>
void device_matrix<float>::cublas_axpy(
    int n, float alpha,
    const float *x, int incx,
    float *y, int incy) {
  CCE(cublasSaxpy(CUBLAS_HANDLE::getInstance(), n, &alpha, x, incx, y, incy));
}

template <>
void device_matrix<double>::cublas_axpy(
    int n, double alpha,
    const double *x, int incx,
    double *y, int incy) {
  CCE(cublasDaxpy(CUBLAS_HANDLE::getInstance(), n, &alpha, x, incx, y, incy));
}

// ++++++++++++++++++++++++++++++++++++++++++++
// +++++ Template Explicit Initialization +++++
// ++++++++++++++++++++++++++++++++++++++++++++
template class device_matrix<float>;
template class device_matrix<double>;
