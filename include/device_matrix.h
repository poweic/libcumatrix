#ifndef __DEVICE_MATRIX_H__
#define __DEVICE_MATRIX_H__

#include <cassert>
#include <string>
using namespace std;

#ifdef HAS_HOST_MATRIX
#include <matrix.h>
#define host_matrix Matrix2D
#endif

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

#define CCE(x) checkCudaErrors(x)
#define STRIDE (sizeof(T) / sizeof(float))

template <typename T>
class SCALAR_MEMORY_BUFFER {
public:
  static T* getBuffer() {
    static SCALAR_MEMORY_BUFFER buffer;
    return buffer._ptr;
  }
private:
  SCALAR_MEMORY_BUFFER(): _ptr(NULL) {
    float scalar = 1;
    CCE(cudaMalloc((void **) &_ptr, sizeof(T)));
    CCE(cudaMemcpy(_ptr, &scalar, sizeof(T), cudaMemcpyHostToDevice));
  };
  SCALAR_MEMORY_BUFFER(const SCALAR_MEMORY_BUFFER& source);
  ~SCALAR_MEMORY_BUFFER() { CCE(cudaFree(_ptr)); }
  SCALAR_MEMORY_BUFFER& operator = (const SCALAR_MEMORY_BUFFER& rhs);

  T* _ptr;
};

class CUBLAS_HANDLE {
public:
  CUBLAS_HANDLE()  { CCE(cublasCreate(&_handle)); }
  ~CUBLAS_HANDLE() { CCE(cublasDestroy(_handle)); }

  cublasHandle_t& get() { return _handle; }
private:
  cublasHandle_t _handle;
};

template <typename T>
class device_matrix {
public:
  // default constructor 
  device_matrix();

  device_matrix(size_t r, size_t c);

  // Load from file. Ex: *.mat in text-form
  device_matrix(const string& filename);

  // Copy Constructor 
  device_matrix(const device_matrix<T>& source);

  // Conversion operator
  operator thrust::device_vector<T>() const;

#ifdef HAS_HOST_MATRIX
  // Constructor from Host Matrix
  device_matrix(const host_matrix<T>& h_matrix);
  
  // Conversion operator
  operator host_matrix<T>() const;
#endif

  // Destructor
  ~device_matrix();

  // ===========================
  // ===== Other Functions =====
  // ===========================
  
  // ===== Addition =====
  device_matrix<T>& operator += (T val);
  device_matrix<T> operator + (T val) const;
  
  device_matrix<T>& operator += (const device_matrix<T>& rhs);
  device_matrix<T> operator + (const device_matrix<T>& rhs) const;

  // ===== Substraction =====
  device_matrix<T>& operator -= (T val);
  device_matrix<T> operator - (T val) const;
  
  device_matrix<T>& operator -= (const device_matrix<T>& rhs);
  device_matrix<T> operator - (const device_matrix<T>& rhs) const;

  // ===== Division =====
  device_matrix<T>& operator /= (T alpha);
  device_matrix<T> operator / (T alpha) const;
  
  // ===== Matrix-scalar Multiplication =====
  device_matrix<T>& operator *= (T alpha);
  device_matrix<T> operator * (T alpha) const;
  // ===== Matrix-Vector Multiplication =====
  // template <typename S>
  // friend thrust::device_vector<S> operator * (const thrust::device_vector<S>& lhs, const device_matrix<S>& m);
  // template <typename S>
  // friend thrust::device_vector<S> operator * (const device_matrix<S>& m, const thrust::device_vector<S>& rhs);
  device_matrix<T> operator * (const thrust::device_vector<T>& rhs) const;

  // ===== Matrix-Matrix Multiplication =====
  device_matrix<T>& operator *= (const device_matrix<T>& rhs);
  device_matrix<T> operator * (const device_matrix<T>& rhs) const;

  template <typename S>
  friend void swap(device_matrix<S>& lhs, device_matrix<S>& rhs);

  template <typename S>
  friend S L1_NORM(const device_matrix<S>& A, const device_matrix<S>& B);

  friend void sgemm(const device_matrix<float>& A, const device_matrix<float>& B, device_matrix<float>& C, float alpha, float beta);

  friend void sgeam(const device_matrix<float>& A, const device_matrix<float>& B, device_matrix<float>& C, float alpha, float beta);

  // Operator Assignment:
  // call copy constructor first, and swap with the temp variable
  device_matrix<T>& operator = (device_matrix<T> rhs);

  void _init();
  void resize(size_t r, size_t c);
  void print(FILE* fid = stdout) const;

  void fillwith(T val) {
    cudaMemset(_data, 0, _rows * _cols * sizeof(T));
  }

  size_t size() const { return _rows * _cols; }
  size_t getRows() const { return _rows; }
  size_t getCols() const { return _cols; }
  T* getData() const { return _data; }
  void save(const string& filename) const;

  static CUBLAS_HANDLE _handle;

private:

  size_t _rows;
  size_t _cols;
  T* _data;
};

template <typename T>
void swap(device_matrix<T>& lhs, device_matrix<T>& rhs) {
  using std::swap;
  swap(lhs._rows, rhs._rows);
  swap(lhs._cols, rhs._cols);
  swap(lhs._data, rhs._data);
}

// In a class template, when performing implicit instantiation, the 
// members are instantiated on demand. Since the code does not use the
// static member, it's not even instantiated in the whole application.
template <typename T>
CUBLAS_HANDLE device_matrix<T>::_handle;

typedef device_matrix<float> dfmat;
typedef thrust::device_vector<float> dfvec;
void sgemm(const dfmat& A, const dfmat& B, dfmat& C, float alpha = 1.0, float beta = 0.0);
void sgeam(const dfmat& A, const dfmat& B, dfmat& C, float alpha = 1.0, float beta = 1.0);
// void saxpy(const dfmat& A, dfmat& B, float alpha = 1.0f);
float snrm2(const dfmat& A);

template <typename T>
device_matrix<T> operator * (T alpha, const device_matrix<T>& m) {
  return m * alpha;
}

template <typename T>
T L1_NORM(const device_matrix<T>& A, const device_matrix<T>& B) {
  return matsum(A - B);
}

#endif // __DEVICE_MATRIX_H__
