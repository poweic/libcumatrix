#include <iostream>
#include <vector>
#include <device_matrix.h>

/* 
 * Add these two headers to support: 
 * (1) Basic File I/O for thrust::device_vector 
 * (2) Matrix - Vector multiplication
 * But you need "nvcc" compiler to compile these two headers
 */
#include <device_arithmetic.h>
#include <device_math.h>
using namespace ext;
using namespace std;

typedef double real;
typedef device_matrix<real> mat;
typedef thrust::device_vector<real> vec;

void randomInit(vec& v);
void randomInit(mat& m);

template <typename T>
struct square {
  __host__ __device__ T operator()(const T& x) const { return x * x; }
};

int main (int argc, char* argv[]) {
  mat A(16, 8);
  randomInit(A);

  vec x(16), y(8), z(8);
  randomInit(x);
  randomInit(y);
  randomInit(z);

  // ==========================================================
  // ===== Converion between std::vector & thrust::vector =====
  // ==========================================================
  std::vector<real> h_x = ext::toStlVector(x);
  x = thrust::device_vector<real>(h_x);
  printf("x = \n"); print(x);

  // =============================
  // ===== Utility Functions =====
  // =============================

  printf("norm(x)   = %.7f \n", norm(x));   // L2-norm
  printf("sum(x)    = %.7f \n\n", sum(x));  // sum

  // ===========================
  // ===== Vector - Scalar =====
  // ===========================

  printf("x + 1.23  = "); print(x + 1.23);
  printf("x - 1.23  = "); print(x - 1.23);
  printf("x * 1.23  = "); print(x * 1.23);
  printf("x / 1.23  = "); print(x / 1.23);

  printf("1.23 + x  = "); print(1.23 + x);
  printf("1.23 - x  = "); print(1.23 - x);
  printf("1.23 * x  = "); print(1.23 * x);
  printf("1.23 / x  = "); print(1.23 / x);

  printf("x += 1.23 = "); print(x += 1.23);
  printf("x -= 1.23 = "); print(x -= 1.23);
  printf("x *= 1.23 = "); print(x *= 1.23);
  printf("x /= 1.23 = "); print(x /= 1.23);

  // ===========================
  // ===== Vector - Vector =====
  // ===========================

  printf("Element-Wise Multiplication: \n");
  printf("y & z = "); print(y & z);

  printf("Multiply to vector to get a matrix: (16 x 1) x (1 x 8) => 16 x 8 \n");
  printf("x * y = \n"); (x * y).print();

  // ===============================
  // ===== Matrix - Vector (1) =====
  // ===============================
  printf("x * A = "); (x * A).print();
  printf("A * y = \n"); (A * y).print();
  printf("x * A * y = "); (x * A * y).print();

  return 0;
}

void randomInit(mat& m) {
  real* h_data = new real [m.size()];
  for (size_t i=0; i<m.size(); ++i)
    h_data[i] = rand() / (real) RAND_MAX;
  cudaMemcpy(m.getData(), h_data, m.size() * sizeof(real), cudaMemcpyHostToDevice);
  delete [] h_data;
}

void randomInit(vec& v) {
  real* h_data = new real [v.size()];
  for (size_t i=0; i<v.size(); ++i)
    h_data[i] = rand() / (real) RAND_MAX;

  v = vec(h_data, h_data + v.size());
  delete [] h_data;
}
