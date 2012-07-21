#include "stdafx.h"
#include "cuda_debug.h"

void cudasafe( cudaError_t error, int device, char* file, int line)
{
  error = cudaGetLastError();
  if(error!=cudaSuccess) { 
    fprintf(stderr,"ERROR: Device: %d: File: %s(%d) : %s(%i)\n",device, file, line, cudaGetErrorString(error), error);  
  }
}