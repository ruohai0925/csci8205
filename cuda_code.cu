extern "C" {
  #include "cuda_code.h"
}


#include <cuda_runtime.h>
// CUDA-C includes
#include <cuda.h>

int dev_id[4];
uint32_t num_dev = 0;

void device_info(){
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("Device %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Total amount of global memory: %.0f MBytes (%llu bytes)\n",(float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
    }
}

void init_gpu_devices(){
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  int dev, driverVersion = 0, runtimeVersion = 0;
  for(dev = 0; dev < deviceCount; ++dev){
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    if((float)deviceProp.totalGlobalMem/1048576.0f > 6000){
      dev_id[num_dev] = dev;
      num_dev++;
      //printf("Device %d: \"%s\"\n", dev, deviceProp.name);
      //printf("  Total amount of global memory: %.0f MBytes (%llu bytes)\n",(float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
    }
    
  }


}