#include <iostream>

#include <device_matrix.h>
#include <device_math_ext.h>
#include <device_arithmetic.h>
#define TEST_CL2E(x, y) {printf("checking "#x" ... "); compareL2error((x), (y));}

using namespace std;

typedef device_matrix<float> mat;
typedef thrust::device_vector<float> vec;

void testing();
void checkErrorIsAcceptable(double err);
void compareL2error(const vec& v, const vec& ref);
void compareL2error(const mat& m, const mat& ref);

int main (int argc, char* argv[]) {
  testing();
  return 0;
}

void testing() {
  printf("Start testing...\n");

  printf("Loading testing data...\n");
  mat A("data/A.mat");
  mat B("data/B.mat");
  mat C("data/C.mat");
  mat D("data/D.mat");

  vec x = ext::load<float>("data/x.vec");
  vec y = ext::load<float>("data/y.vec");
  vec u = ext::load<float>("data/u.vec");
  vec v = ext::load<float>("data/v.vec");

  mat ApB("data/A+B.mat");
  mat AmB("data/A-B.mat");
  mat CmD("data/C-D.mat");
  mat CpD("data/C+D.mat");

  mat AC("data/AC.mat");
  mat AD("data/AD.mat");
  mat BC("data/BC.mat");
  mat BD("data/BD.mat");

  vec Ax = ext::load<float>("data/Ax.vec");
  vec Bx = ext::load<float>("data/Bx.vec");
  vec Cy = ext::load<float>("data/Cy.vec");
  vec Dy = ext::load<float>("data/Dy.vec");

  mat uA("data/uA.vec");
  mat uB("data/uB.vec");
  mat vC("data/vC.vec");
  mat vD("data/vD.vec");

  mat ApPI("data/A+pi.mat");
  mat BmPI("data/B-pi.mat");
  mat eC("data/eC.mat");
  mat D_over_e("data/D_over_e.mat");

  mat uAx("data/uAx.scalar");
  mat uBx("data/uBx.scalar");
  mat vCy("data/vCy.scalar");
  mat vDy("data/vDy.scalar");

  mat xu("data/xu.mat");
  mat xv("data/xv.mat");
  mat yu("data/yu.mat");
  mat yv("data/yv.mat");

  printf("\n===== FILE I/O =====\n");
  A.save("/tmp/cumatrix.mat");
  TEST_CL2E(mat("/tmp/cumatrix.mat"), A)

  printf("\n===== Vector - Vector Multiplication =====\n");
  TEST_CL2E(x * u, xu);
  TEST_CL2E(x * v, xv);
  TEST_CL2E(y * u, yu);
  TEST_CL2E(y * v, yv);

  printf("\n===== FILE I/O Testing =====\n");
  A.save("/tmp/cumatrix.mat");
  TEST_CL2E(mat("/tmp/cumatrix.mat"), A)

  printf("\n===== Matrix Addition =====\n");
  TEST_CL2E(A + B, ApB);
  TEST_CL2E(A - B, AmB);
  TEST_CL2E(C + D, CpD);
  TEST_CL2E(C - D, CmD);

  printf("\n===== Matrix - Scalar Arithmetic =====\n");
  TEST_CL2E(A + PI, ApPI);
  TEST_CL2E(B - PI, BmPI);
  TEST_CL2E(2.718281828f * C, eC);
  TEST_CL2E(D / 2.718281828f, D_over_e);

  printf("\n===== Matrix - Matrix Multiplication =====\n");
  TEST_CL2E(A * C, AC);
  TEST_CL2E(A * D, AD);
  TEST_CL2E(B * C, BC);
  TEST_CL2E(B * D, BD);

  printf("\n===== Matrix - Vector Multiplication (1) =====\n");
  TEST_CL2E(A * x, Ax);
  TEST_CL2E(B * x, Bx);
  TEST_CL2E(C * y, Cy);
  TEST_CL2E(D * y, Dy);

  TEST_CL2E(u * A, uA);
  TEST_CL2E(u * B, uB);
  TEST_CL2E(v * C, vC);
  TEST_CL2E(v * D, vD);

  printf("\n===== Matrix - Vector Multiplication (2) =====\n");
  TEST_CL2E(u * A * x, uAx);
  TEST_CL2E(u * B * x, uBx);
  TEST_CL2E(v * C * y, vCy);
  TEST_CL2E(v * D * y, vDy);

  printf("\n[Done]\n");
}

void compareL2error(const mat& m, const mat& ref) {
  float error = snrm2(m - ref) / snrm2(ref);
  checkErrorIsAcceptable(error);
}

void compareL2error(const vec& v, const vec& ref) {
  float error = norm(v - ref) / norm(ref);
  checkErrorIsAcceptable(error);
}

void checkErrorIsAcceptable(double error) {
  const float EPS = 1e-6;
  if (error < EPS)
    printf("\33[32m[ OK ]\33[0m \n");
  else
    printf("error = %.4e > EPS (%.4e) \33[31m[FAILED]\33[0m\n", error, EPS);
}
