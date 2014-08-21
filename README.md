libcumatrix
======

# C++ Interface of CUDA-based GPU matrix library.

# Prerequisite
You need
- An NVIDIA Graphic Processing Unit (GPU) that supports CUDA (Ex: GTX-660)
- Linux / Ubuntu
- Install [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (at least CUDA 5.0)

# Quick Start
Just type "make" and run the following programs:
1) ./benchmark
2) ./example1
3) ./example2

# How to Use It in My C++ Codes
Include device_matrix.h and add -I flags to your g++ command


## Here's how you use it in your program.
```c++
#include <device_matrix.h>

int main() {
  // create a 10 by 20 matrix lies in GPU memory (filled with 0)
  device_matrix<float> A(10, 20, 0);
  A.print();

  A += 10;
  A.print();
}

```
## Here's how you add -I flags and libraries in your g++ command

On linking, libcumatrix needs:
1) -lcuda   (libcuda.so  )
2) -lcublas (libcublas.so)
3) -lcudart (libcudart.so)

So
```makefile
  g++ -o my_program main.cpp -I libcumatrix/include -lcuda -lcublas -lcudart -lcumatrix
```

By default, the root directories containing CUDA is located at "/usr/local/cuda". If you
wish to change it, open the "Makefile" and change CUDA_ROOT to where it lies.

# Others
For an older version of CUDA sdk, Thrust Library did not lie directly in "<path-to-cuda>/include/thrust" (something like <path-to-cuda>/include/thrust-1.5.1). Create a symbolic link for it using the following command:
$ ln -s /usr/local/cuda/include/thrust-1.5.1 /usr/local/cuda/include/thrust
