#include <device_matrix.h>
#define mylog(token) {cout << #token " = " << token << endl;}

// ===============================
// ===== class device_matrix =====
// ===============================

template <typename T>
device_matrix<T>::device_matrix(): _rows(0), _cols(0), _data(NULL) { }

template <typename T>
device_matrix<T>::device_matrix(size_t r, size_t c): _rows(r), _cols(c), _data(NULL) {
  _init();
  fillwith(0);
}

template <typename T>
device_matrix<T>::device_matrix(const string& filename): _rows(0), _cols(0), _data(NULL) {

  const size_t MAX_BUFFER = 65536;
  char line[MAX_BUFFER];

  FILE* fid = fopen(filename.c_str(), "r");
  while (fgets(line, MAX_BUFFER, fid)) {
    _rows++;

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
  T* data = new T[_rows*_cols];
  for (size_t i=0; i<_rows; ++i)
    for (size_t j=0; j<_cols; ++j)
      fscanf(fid, "%f ", &(data[j*_rows + i]));
  fclose(fid);

  _init();
  CCE(cudaMemcpy(_data, data, sizeof(T) * _rows * _cols, cudaMemcpyHostToDevice));
  delete [] data;
}
// Copy Constructor 
template <typename T>
device_matrix<T>::device_matrix(const device_matrix<T>& source): _rows(source._rows), _cols(source._cols), _data(NULL) {
  _init();
  CCE(cudaMemcpy(_data, source._data, sizeof(T) * _rows * _cols, cudaMemcpyDeviceToDevice));
}

// Conversion operator
template <typename T>
device_matrix<T>::operator thrust::device_vector<T>() const {
  assert(_rows == 1 || _cols == 1);
  return thrust::device_vector<T>(_data, _data + size());
}

#ifdef HAS_HOST_MATRIX
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
  CCE(cublasSaxpy(CUBLAS_HANDLE::getInstance(), _rows*_cols, &val, SCALAR_MEMORY_BUFFER<T>::getBuffer(), 0, _data, 1));
  return *this;
} 

template <typename T>
device_matrix<T> device_matrix<T>::operator + (T val) const {
  device_matrix<T> m(*this);
  return (m += val);
}

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
template <typename T>
device_matrix<T>& device_matrix<T>::operator -= (T val) {
  val = -val;
  CCE(cublasSaxpy(CUBLAS_HANDLE::getInstance(), _rows*_cols, &val, SCALAR_MEMORY_BUFFER<T>::getBuffer(), 0, _data, 1));
  return *this;
}

template <typename T>
device_matrix<T> device_matrix<T>::operator - (T val) const {
  device_matrix<T> m(*this);
  return (m -= val);
}

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
  status = cublasSscal(CUBLAS_HANDLE::getInstance(), _rows*_cols, &alpha, _data, STRIDE);
  CCE(status);
  return *this;
}

template <typename T>
device_matrix<T> device_matrix<T>::operator * (T alpha) const {
  device_matrix<T> result(*this);
  return result *= alpha;
}

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
void device_matrix<T>::print(FILE* fid) const {

  T* data = new T[size()];
  CCE(cudaMemcpy(data, _data, sizeof(T) * size(), cudaMemcpyDeviceToHost));

  for (size_t i=0; i<_rows; ++i) {
    for (size_t j=0; j<_cols; ++j)
      fprintf(fid, "%.7f ", data[j*_rows + i]);
    fprintf(fid, "\n");
  }

  if (fid == stdout)
    fprintf(fid, "\n");

  delete [] data;
}

template <typename T>
void device_matrix<T>::fillwith(T val) {
  cudaMemset(_data, 0, _rows * _cols * sizeof(T));
}

template <typename T>
void device_matrix<T>::save(const string& filename) const {
  FILE* fid = fopen(filename.c_str(), "w");
  if (fid == NULL)
    return;

  print(fid);
  fclose(fid);
}
// ++++++++++++++++++++++++++++++++++++++++++++
// +++++ Template Explicit Initialization +++++
// ++++++++++++++++++++++++++++++++++++++++++++
template class device_matrix<float>;

float snrm2(const dfmat& A) {
  float result;
  cublasStatus_t status;
  status = cublasSnrm2(CUBLAS_HANDLE::getInstance(), A.size(), A.getData(), 1, &result);
  CCE(status);
  return result;
}

void sgemm(const dfmat& A, const dfmat& B, dfmat& C, float alpha, float beta) {
  // Perform C = αA*B + βC, not transpose on A and B
  size_t m = A._rows;
  size_t n = B._cols;
  C.resize(m, n);

  size_t k = A._cols;

  int lda = A._rows;
  int ldb = B._rows;
  int ldc = C._rows;

  cublasStatus_t status;
  status = cublasSgemm(CUBLAS_HANDLE::getInstance(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A._data, lda, B._data, ldb, &beta, C._data, ldc);

  CCE(status);
}

void sgeam(const dfmat& A, const dfmat& B, dfmat& C, float alpha, float beta) {
  // Perform C = αA + βB, not transpose on A and B
  assert(A._rows == B._rows && A._cols == B._cols);
  
  size_t m = A._rows;
  size_t n = A._cols;
  C.resize(m, n);

  int lda = A._rows;
  int ldb = B._rows;
  int ldc = C._rows;

  cublasStatus_t status;
  status = cublasSgeam(CUBLAS_HANDLE::getInstance(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha, A._data, lda, &beta, B._data, ldb, C._data, ldc);
  CCE(status);
}
