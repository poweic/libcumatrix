#include <iostream>
#include <color.h>

#include <device_matrix.h>

using namespace std;

typedef device_matrix<float> mat;

struct Timer {
  Timer();
  void tic();
  float toc();
  cudaEvent_t start, stop;
};

void showGFlops(double flops, float time);
void testing();
void compareL2error(const mat& m, const mat& ref);
void benchmark();

int main (int argc, char* argv[]) {

  benchmark();

  return 0;
}

void testing() {
  mat A("data/A.mat");
  mat B("data/B.mat");
  mat C("data/C.mat");

  compareL2error(A*B, C);
}

void compareL2error(const mat& m, const mat& ref) {
  float error = snrm2(m - ref) / snrm2(ref);
  printf("error = %.7e \n", error);
}

void benchmark() {

  mat A("data/A.mat");
  mat B("data/B.mat");
  mat C;

  Timer timer;
  timer.tic();

  int nIter = 128;
  for (int i=0; i<nIter; ++i)
    sgemm(A, B, C);

  float avgTime = timer.toc() / nIter;
  double flops = 2.0 * (double) A.getRows() * (double) A.getCols() * (double) B.getCols();
  showGFlops(flops, avgTime);

  Timer timer2;
  timer2.tic();
  for (int i=0; i<nIter; ++i)
    mat C(A * B);

  avgTime = timer2.toc() / nIter;
  showGFlops(flops, avgTime);
}

void showGFlops(double flops, float time) {
  double gigaFlops = (flops * 1.0e-9f) / (time / 1000.0f);
  printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n", gigaFlops, time, flops);
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

