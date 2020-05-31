#ifndef CUDA_CODE_H
#define CUDA_CODE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


extern int dev_id[4];
extern uint32_t num_dev;

void device_info();

void init_gpu_devices();

#endif /* CUDA_CODE_H */