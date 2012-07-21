#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"

#define CS(device, x) cudasafe(x, device, __FILE__, __LINE__)
void cudasafe( cudaError_t error, int device, char* file, int line);