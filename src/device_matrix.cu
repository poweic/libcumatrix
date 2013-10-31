#include <device_matrix.h>
#define mylog(token) {cout << #token " = " << token << endl;}
#include <blas.h>
#include <device_blas.h>

template <typename T>
device_matrix<T>::device_matrix(): _rows(0), _cols(0), _data(NULL) { }

template <typename T>
device_matrix<T>::device_matrix(size_t r, size_t c): _rows(r), _cols(c), _data(NULL) {
  _init();
  fillwith(0);
}

template <typename T>
device_matrix<T>::device_matrix(const string& filename): _rows(0), _cols(0), _data(NULL) {
  host_matrix<T> hm(filename);
  *this = device_matrix<T>(hm);
}
// Copy Constructor 
template <typename T>
device_matrix<T>::device_matrix(const device_matrix<T>& source): _rows(source._rows), _cols(source._cols), _data(NULL) {
  _init();

  CCE(cudaMemcpy(_data, source._data, sizeof(T) * _rows * _cols, cudaMemcpyHostToHost));
}

// Constructor from Host Matrix
template <typename T>
device_matrix<T>::device_matrix(const host_matrix<T>& h_matrix): _rows(h_matrix.getRows()), _cols(h_matrix.getCols()), _data(NULL) {

  // Convert T** to column major using transpose
  host_matrix<T> cm_h_matrix(~h_matrix);
  _init();

  size_t n = _rows * _cols;

  T* h_data = new T[n];
  for (size_t i=0; i<_cols; ++i)
    memcpy(h_data + i*_rows, cm_h_matrix[i], sizeof(T) * _rows);

  CCE(cudaMemcpy(_data, h_data, sizeof(T) * _rows * _cols, cudaMemcpyHostToDevice));
  // CCE(cublasSetVector(n, sizeof(T), h_data, STRIDE, _data, STRIDE));

  delete [] h_data;
}

template <typename T>
device_matrix<T>::~device_matrix() {
  CCE(cudaFree(_data));
}

// ===========================
// ===== Other Functions =====
// ===========================

// ===== Addition =====
// device_matrix<T>& operator += (T val) { return *this; } 
// device_matrix<T> operator + (T val) const { return *this; }

template <typename T>
device_matrix<T>& device_matrix<T>::operator += (const device_matrix<T>& rhs) {
  *this = *this + rhs;
  return *this;
}

template <typename T>
device_matrix<T> device_matrix<T>::operator + (const device_matrix<T>& rhs) const {
  device_matrix<T> result(_rows, _cols);
  sgeam(*this, rhs, result, 1.0, 1.0);
  return result;
}

// ===== Substraction =====
// device_matrix<T>& operator -= (T val) { return *this; }
// device_matrix<T> operator - (T val) const { return *this; }

template <typename T>
device_matrix<T>& device_matrix<T>::operator -= (const device_matrix<T>& rhs) {
  *this = *this - rhs;
  return *this;
}

template <typename T>
device_matrix<T> device_matrix<T>::operator - (const device_matrix<T>& rhs) const {
  device_matrix<T> result(_rows, _cols);
  sgeam(*this, rhs, result, 1.0, -1.0);
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
  cublasStatus_t status;
  status = cublasSscal(device_matrix<float>::_handle.get(), _rows*_cols, &alpha, _data, STRIDE);
  CCE(status);
  return *this;
}

template <typename T>
device_matrix<T> device_matrix<T>::operator * (T alpha) const {
  device_matrix<T> result(*this);
  return result *= alpha;
}

// ===== Matrix-Vector Multiplication =====
/*template <typename T>
thrust::device_vector<T> device_matrix<T>::operator * (const thrust::device_vector<T>& rhs) const {
  assert(_cols == rhs.size());
  thrust::device_vector<T> result(_rows);
  return result;
}*/

// ===== Matrix-Matrix Multiplication =====
template <typename T>
device_matrix<T>& device_matrix<T>::operator *= (const device_matrix<T>& rhs) {
  *this = *this * rhs;
  return *this;
}

template <typename T>
device_matrix<T> device_matrix<T>::operator * (const device_matrix<T>& rhs) const {
  device_matrix<T> result(_rows, rhs._cols);
  sgemm(*this, rhs, result);
  return result;
}

// ==================================================
template <typename T>
device_matrix<T>::operator host_matrix<T>() const {

  host_matrix<T> cm_h_matrix(_cols, _rows);

  size_t n = _rows * _cols;
  T* h_data = new T[n];

  CCE(cublasGetVector(n, sizeof(T), _data, STRIDE, h_data, STRIDE));

  for (size_t i=0; i<_cols; ++i)
    memcpy(cm_h_matrix[i], h_data + i*_rows, sizeof(T) * _rows);

  delete [] h_data;

  return ~cm_h_matrix;
}

// Operator Assignment:
// call copy constructor first, and swap with the temp variable
template <typename T>
device_matrix<T>& device_matrix<T>::operator = (device_matrix<T> rhs) {
  swap(*this, rhs);
  return *this;
}

template <typename T>
void device_matrix<T>::_init() {
  CCE(cudaMalloc((void **)&_data, _rows * _cols * sizeof(T)));
}

template <typename T>
void device_matrix<T>::resize(size_t r, size_t c) {
  if (_rows == r && _cols == c)
    return;

  _rows = r;
  _cols = c;
  _init();
  fillwith(0);
}

template <typename T>
void device_matrix<T>::print(size_t precision) const {
  host_matrix<T> hm(*this);
  hm.print(precision);
}

template <typename T>
void device_matrix<T>::saveas(const string& filename) const {
  host_matrix<T> hm(*this);
  hm.saveas(filename);
}
// ++++++++++++++++++++++++++++++++++++++++++++
// +++++ Template Explicit Initialization +++++
// ++++++++++++++++++++++++++++++++++++++++++++
template class device_matrix<float>;

float snrm2(const dmat& A) {
  float result;
  cublasStatus_t status;
  status = cublasSnrm2(dmat::_handle.get(), A.size(), A.getData(), 1, &result);
  CCE(status);
  return result;
}

void sgemm(const dmat& A, const dmat& B, dmat& C, float alpha, float beta) {
  // Perform C = αA*B + βC, not transpose on A and B
  size_t m = A._rows;
  size_t n = B._cols;
  C.resize(m, n);

  size_t k = A._cols;

  int lda = A._rows;
  int ldb = B._rows;
  int ldc = C._rows;

  cublasStatus_t status;
  status = cublasSgemm(dmat::_handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A._data, lda, B._data, ldb, &beta, C._data, ldc);

  CCE(status);
}

void sgeam(const dmat& A, const dmat& B, dmat& C, float alpha, float beta) {
  // Perform C = αA + βB, not transpose on A and B
  assert(A._rows == B._rows && A._cols == B._cols);
  
  size_t m = A._rows;
  size_t n = A._cols;
  C.resize(m, n);

  int lda = A._rows;
  int ldb = B._rows;
  int ldc = C._rows;

  cublasStatus_t status;
  status = cublasSgeam(dmat::_handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha, A._data, lda, &beta, B._data, ldb, C._data, ldc);
  CCE(status);
}
