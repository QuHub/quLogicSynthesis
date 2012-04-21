#pragma  once

// Approximate Max size per Device: 1.5GByte
// Controls: 4 * MAX_GATES * NUMBER_OF_CUDA_BLOCKS
// Target+Ops: 2 * MAX_GATES * NUMBER_OF_CUDA_BLOCKS
// NGates: 4 * NUMBER_OF_CUDA_BLOCKS 
#define by(x) (x*sizeof(int))
#define NUMBER_OF_CUDA_BLOCKS 512
#define MAX_GATES  400*1024        
