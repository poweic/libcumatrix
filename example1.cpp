#include <iostream>
#include <device_matrix.h>

using namespace std;
typedef device_matrix<float> mat;

void randomInit(mat& m);

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
  printf("3.14 + A\n"); (3.14 + A).print();
  printf("1.23 * A\n"); (1.23 * A).print();
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

  return 0;
}

void randomInit(mat& m) {
  float* h_data = new float [m.size()];
  for (int i=0; i<m.size(); ++i)
    h_data[i] = rand() / (float) RAND_MAX;
  cudaMemcpy(m.getData(), h_data, m.size() * sizeof(float), cudaMemcpyHostToDevice);
  delete [] h_data;
}
