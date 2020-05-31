#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
 

#define N_DIM 1000

/* CUDA kernel parameters */
#define NUM_BLOCKS  1
#define NUM_THREADS 32

// CUDA kernel to add elements of two arrays
__global__
void add(unsigned long int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (unsigned long int i = index; i < n; i += stride){
    y[i] = x[i] + y[i];
    //if(i%100000==0 && i > 480000000)printf("Adding i=%lu\n", i);
  }
}
 
int main(void)
{
  uint64_t n = N_DIM, N=n*n;

  double *func_values;


  // Allocate pinned memory
  cudaMallocHost((void **) &func_values, N*sizeof(double));
 
  printf("Writing 1.0 to function values\n");

  // initialize x and y arrays on the host
  for (uint64_t i = 0; i < N; i++) {
    func_values[i] = 1.0;
  }
 
  printf("Finished writing values\n");


  // Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = 1;
  //add<<<numBlocks, blockSize>>>(N, x, y);
 
  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();
 

  cudaFreeHost(func_values);
  return 0;
}
