#ifndef __CUDACC__
#pragma message "\33[33mPotentially wrong compiler. Please use nvcc instead \33[0m"
#endif

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#define HAVE_THRUST_DEVICE_VECTOR_H 1

// =====================================
// ===== Vector - Scalar Operators =====
// =====================================
#define VECTOR thrust::device_vector
#define WHERE thrust
#include <functional.inl>
#include <operators.inl>
#undef VECTOR
#undef WHERE
