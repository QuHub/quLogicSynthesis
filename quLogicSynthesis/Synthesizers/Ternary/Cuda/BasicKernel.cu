#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaSequence.h"
#include "stdio.h"
#include "cuda_debug.h"


#define BIT(x,i) ((x & gcuBitMask[i]) >> 2*i)

// Ternary Gates
int gTernaryOps[5][3]= 
{
  {2, 0, 1},      // 0: -1
  {1, 2, 0},      // 1: -2
  {1, 0, 2},      // 2: 01
  {2, 1, 0},      // 3: 02
  {0, 2, 1}       // 4: 12
};                

// Operation to use based on [input][output] values which are an index to the gates in the m_Op array above
// Example: 
//    Input = 2, Output = 1 => Gate 4 (From m_Op above would be Swap gate 12)
int gOpMap[3][3] =
{
  // Output    0, 1, 2     Input 
  {4, 2, 3}, // 0
  {2, 3, 4}, // 1
  {3, 4, 2}  // 2
};

// Bitmask two bits at a time for ternary operations.
int gBitMask[] = {3, 3<<2, 3<<4, 3<<6, 3<<8, 3<<10, 3<<12, 3<<14, 3<<16}; 
__device__ __constant__ int gcuBitMask[sizeof(gBitMask)];
__device__ __constant__ int gcuTernaryOps[5][3];
__device__ __constant__ int gcuOpMap[3][3];

__device__ void Process(int inTerm, int outTerm, int nBits, PINT gBitMask, PINT pTarget, PINT pControl, PINT pOperation);


__global__ void cuSynthesizeKernel(CudaSequence *data)
{
  CudaSequence seq = data[0];
  int index = blockIdx.x * seq.m_outputBlockSize;
  seq.m_cuGates[index] = 0;

  for(int i=0; i<seq.m_nTerms; i++) {
    Process(seq.m_cuIn[index+i], 
      seq.m_cuOut[index+i], 
      seq.m_nBits,
      &seq.m_cuGates[index],
      &seq.m_cuTarget[index], 
      &seq.m_cuControl[index],
      &seq.m_cuOperation[index]
    );
  }

  //for(int i=0; i < 10; i++) {
  //  target[index+i] = 0x20;
  //  control[index+i] = index*100;
  //}

}

void SynthesizeKernel(CudaSequence *pcuSeq)
{
  // Constants are scoped to a file, and cannot use extern..
  CS( cudaMemcpyToSymbol(gcuBitMask, gBitMask, sizeof(gBitMask)) );
  CS( cudaMemcpyToSymbol(gcuTernaryOps, gTernaryOps, sizeof(gTernaryOps)) );
  CS( cudaMemcpyToSymbol(gcuOpMap, gOpMap, sizeof(gOpMap)) );
  cuSynthesizeKernel<<<1, 1>>>(pcuSeq);
}

__device__ int Propagate(int outTerm, PINT pTarget, PINT pOperation, PINT pControl, int nGates)
{
  // Apply current list of gates..
  for (int i=0; i<nGates; i++) {
    int mask = gcuBitMask[pTarget[i]];
    if ( pControl[i] == (~mask & outTerm) ) {               // Control Bits for gate matches All bits in output excluding target bits.
      int val = (mask & outTerm) >> 2*pTarget[i];           // Bring target bits to lower two bits.
      val = (gcuTernaryOps[pOperation[i]][val] << 2*pTarget[i]);       // Apply operation on bits.
      outTerm = (~mask & outTerm) | val;
    }
  }

  return outTerm;
}

__device__ void Process(int inTerm, int outTerm, int nBits, PINT pnGates, PINT pTarget, PINT pControl, PINT pOperation)
{
  printf("\n****** In,out:[%d, %d] ", inTerm, outTerm);
  outTerm = Propagate(outTerm, pTarget, pOperation, pControl, *pnGates);

  printf("After Propgate: %d \n", outTerm);

  //  process low (output) to high (input) transitions first then high to low
  for(int dir=1; dir>-2; dir-=2) {
    for (int i=0; i < nBits; i++) {
      // Isloate bit (i) for processing
      int inBit  = (gcuBitMask[i] & inTerm);        // must be defined as signed int
      int outBit = (gcuBitMask[i] & outTerm);

      if ( dir * (inBit - outBit) > 0) {         // Difference? Yes!
        pTarget   [*pnGates] = i;                           // Save index of target bits
        pControl  [*pnGates] = ~gcuBitMask[i] & outTerm;      // For now, it is everything except target bits is a control bit
        pOperation[*pnGates] = gcuOpMap[BIT(inTerm,i)][BIT(outTerm,i)];  // Find the appropriate operation. 
        printf("dir(%d) [C,T,O] [%d, %d, %d] ", dir, pControl[*pnGates], i, pOperation[*pnGates]);
        (*pnGates)++;
        outTerm = (~gcuBitMask[i] & outTerm) | (gcuBitMask[i] & inTerm);
        printf(" => %d \n", outTerm);
      }
    }
  }
}


