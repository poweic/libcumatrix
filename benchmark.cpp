#include <iostream>

#include <device_matrix.h>
using namespace std;

typedef device_matrix<float> mat;

struct Timer {
  Timer();
  void tic();
  float toc();
  cudaEvent_t start, stop;
};

template <typename T>
void randomInit(device_matrix<T>& m) {
  T* h_data = new T [m.size()];
  for (int i=0; i<m.size(); ++i)
    h_data[i] = rand() / (T) RAND_MAX;
  cudaMemcpy(m.getData(), h_data, m.size() * sizeof(T), cudaMemcpyHostToDevice);
  delete [] h_data;
}

void benchmark();
void matrixMul(int m, int n, int l);
void showGFlops(double flops, float time);

int main (int argc, char* argv[]) {
  benchmark();
  return 0;
}

void benchmark() {
  srand(2013);
  matrixMul(32, 48, 16);
  matrixMul(64, 96, 32);
  matrixMul(128, 192, 64);
  matrixMul(256, 384, 128);
  matrixMul(512, 768, 256);
  matrixMul(1024, 1536, 512);
  matrixMul(2048, 3072, 1024);
}

void matrixMul(int m, int n, int l) {

  mat A(m, n), B(n, l), C;
  randomInit(A);
  randomInit(B);

  int nIter = 128;
  Timer timer1, timer2;

  // ===== Method 1 : sgemm(A,B,C) =====
  timer1.tic();
  for (int i=0; i<nIter; ++i)
    gemm(A, B, C);
  float avgTime1 = timer1.toc() / nIter;

  // ===== Method 2 : C = A*B =====
  timer2.tic();
  for (int i=0; i<nIter; ++i)
    mat C(A * B);
  float avgTime2 = timer2.toc() / nIter;

  // ===== Calculate GFlops =====
  double flops = 2.0 * (double) A.getRows() * (double) A.getCols() * (double) B.getCols();
  double gigaFlops1 = (flops * 1.0e-9f) / (avgTime1 / 1000.0f);
  double gigaFlops2 = (flops * 1.0e-9f) / (avgTime2 / 1000.0f);

  // ===== Benchmark Summary =====
  printf("            Matrix Multiplication         \n"
         "+----------------------------------------+\n"
         "|   matrix         rows         cols     |\n"
	 "+----------------------------------------+\n"
	 "|      A           %4lu         %4lu     |\n"
	 "|                                        |\n"
	 "|      B           %4lu         %4lu     |\n"
	 "|                                        |\n"
	 "|     AxB          %4lu         %4lu     |\n"
	 "+----------------------------------------+\n"
	 "| Performance   sgemm(A,B,C)   C = A*B   |\n"
	 "|   GFlops        %5.1f        %5.1f     |\n"
	 "+----------------------------------------+\n\n",
	 A.getRows(), A.getCols(),
	 B.getRows(), B.getCols(),
	 A.getRows(), B.getCols(),
	 gigaFlops1, gigaFlops2);
}

Timer::Timer() {
  CCE(cudaEventCreate(&start));
  CCE(cudaEventCreate(&stop));
}

void Timer::tic() {
  CCE(cudaEventRecord(start, NULL));
}

float Timer::toc() {
  CCE(cudaEventRecord(stop, NULL));
  CCE(cudaEventSynchronize(stop));

  float diff = 0.0f;
  CCE(cudaEventElapsedTime(&diff , start, stop));
}
