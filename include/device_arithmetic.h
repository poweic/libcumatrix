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
template <typename T> T norm(const thrust::host_vector<T>& v);
template <typename T> T norm(const thrust::device_vector<T>& v);
template <typename T> void print(const thrust::host_vector<T>& v);
template <typename T> void print(const thrust::device_vector<T>& v);
template <typename T> dmat<T> operator & (const dvec<T>& v, const dmat<T>& m);

// =====================================
// ===== Matrix - Vector Operators =====
// =====================================

template <typename T> dmat<T> operator * (const dvec<T>& col_vector, const dvec<T>& row_vector);
template <typename T> dvec<T> operator & (const dvec<T>& x, const dvec<T>& y);
template <typename T> dmat<T> operator * (const dvec<T>& v, const dmat<T>& m);
template <typename T> dmat<T> operator * (const dmat<T>& m, const dvec<T>& v);

#undef dvec
#undef dmat

#endif // __DEVICE_BLAS_H_
