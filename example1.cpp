#include <iostream>
#include <vector>
#include <device_matrix.h>

using namespace std;
typedef device_matrix<float> mat;

template <typename T>
void randomInit(device_matrix<T>& m) {
  T* h_data = new T [m.size()];
  for (int i=0; i<m.size(); ++i)
    h_data[i] = rand() / (T) RAND_MAX;
  cudaMemcpy(m.getData(), h_data, m.size() * sizeof(T), cudaMemcpyHostToDevice);
  delete [] h_data;
}

int main (int argc, char* argv[]) {

  mat A(16, 16), B(16, 16);
  randomInit(A);
  randomInit(B);

  // ============================
  // ===== Print & FILE I/O =====
  // ============================
  A.print();
  B.print();


  // To save matrix, you can pass a file descriptor (fid) to print()
  FILE* fid = fopen("A.mat", "w");
  if (fid != NULL)
    A.print(fid);
  fclose(fid);

  // Or you can just call A.save();  It's more simple !
  A.save("A.mat");

  // ======================================
  // ===== Construct from CPU pointer =====
  // ======================================

  int M = 12, N = 17;
  float* x = new float[M * N];

  for (int i=0; i<M*N; ++i)
    x[i] = i;

  mat(x, M, N).print();

  // ===============================
  // ===== Matrix - Scalar (1) =====
  // ===============================
  printf("A + 3.14\n"); (A + 3.14).print();
  printf("A - 3.14\n"); (A - 3.14).print();
  printf("A * 1.23\n"); (A * 1.23).print();
  printf("A / 1.23\n"); (A / 1.23).print();

  // ===============================
  // ===== Matrix - Scalar (2) =====
  // ===============================
  printf("3.14 + A\n"); (3.14f + A).print();
  printf("1.23 * A\n"); (1.23f * A).print();
  // printf("3.14 - A\n"); (3.14 - A).print(); [NOT IMPLEMENTED YET]
  // printf("1.23 / A\n"); (1.23 / A).print(); [NOT IMPLEMENTED YET]
  
  // ===============================
  // ===== Matrix - Scalar (3) =====
  // ===============================
  printf("A += 3.14\n"); (A += 3.14).print();
  printf("A -= 3.14\n"); (A -= 3.14).print();
  printf("A *= 1.23\n"); (A *= 1.23).print();
  printf("A /= 1.23\n"); (A /= 1.23).print();

  // ===========================
  // ===== Matrix - Matrix =====
  // ===========================
  printf("A+B:\n"); (A+B).print();
  printf("A-B:\n"); (A-B).print();
  printf("A*B:\n"); (A*B).print();
  // printf("A/B:\n"); (A/B).print(); [NOT IMPLEMENTED YET]

  mat C(12, 8), D(8, 12), E(12, 8);
  randomInit(C);
  randomInit(D);
  randomInit(E);
  printf("C:\n"); C.print();
  printf("D:\n"); D.print();
  printf("E:\n"); E.print();

  printf("C + ~D:\n"); (C + ~D).print();
  printf("~C + D:\n"); (~C + D).print();
  
  printf("C - ~D:\n"); (C - ~D).print();
  printf("~C - D:\n"); (~C - D).print();

  printf("C * D: \n"); (C*D).print();
  printf("~C * ~D:\n"); (~C * ~D).print();

  printf("C * ~E:\n"); (C * ~E).print();
  printf("~C * E:\n"); (~C * E).print();

  printf("C *= ~E:\n"); (C *= ~E).print();
  printf("C += ~C:\n"); (C += ~C).print();
  printf("C -= ~C:\n"); (C -= ~C).print();

  // ==============================================================

  mat E1(10, 10);
  mat::cublas_gemm(CUBLAS_OP_N, CUBLAS_OP_N, 10, 10, 8, 1.0, E.getData(), E.getRows(), D.getData(), D.getRows(), 0, E1.getData(), E1.getRows());

  printf("E1 = E(1:10, :) * D(:, 1:10)\n");
  E1.print();

  mat E2(12, 11);
  mat::cublas_geam(CUBLAS_OP_N, CUBLAS_OP_N, 10, 10, 1.0, E1.getData(), 10, 1.0, E2.getData(), E2.getRows(), E2.getData(), E2.getRows());
  printf("E2: \n");
  E2.print();

  E2.resize(12, 3);
  printf("E2.resize(5, 3): \n");
  E2.print();

  E2.fillwith(123);
  printf("E2.fillwith(5): \n");
  E2.print();

  return 0;
}

