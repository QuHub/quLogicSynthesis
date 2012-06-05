#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaSequence.h"
#include "stdio.h"
#include "cuda_debug.h"
#include "../../../constants.h"


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

__device__ void Process(int inTerm, int outTerm, int nBits, PINT gBitMask, PINT pControl, PBYTE pTarget, PBYTE pOperation);


__device__ void CopySharedToGlobal(PINT pDst, PINT pSrc, int nWords)
{
  for (int i=0; i<nWords; i++)
    pDst[i] = pSrc[i];
}

__device__ void CopySharedBytesToGlobal(PBYTE pDst, PBYTE pSrc, int nBytes)
{
  for (int i=0; i<nBytes; i++)
    pDst[i] = pSrc[i];
}

__global__ void cuSynthesizeKernel(CudaSequence *data)
{
  CudaSequence seq = data[0];
  int inputIndex =  threadIdx.x * seq.m_nTerms; 
  int outputIndex = threadIdx.x * seq.m_nMaxGates;
  int nGates = 0;
  int nBits = seq.m_nBits;

  __shared__ int pIn[243];
  __shared__ int pOut[243];
  __shared__ int pControl[3*1024];
  __shared__ BYTE pGates[3*1024];
  __shared__ BYTE pTarget[3*1024];

  for(int i=0; i<seq.m_nTerms; i++) {
    pIn[i] = seq.m_cuIn[inputIndex+i]; 
    pOut[i] = seq.m_cuOut[inputIndex+i]; 
  }

  for(int i=0; i<seq.m_nTerms; i++) {
    Process(pIn[i], 
      pOut[i], 
      nBits,
      &nGates,
      pControl,
      pTarget,
      pGates
    );
  }

  __syncthreads();

  for(int i=0; i<nGates; i++) {
    CopySharedToGlobal(&seq.m_cuControl[outputIndex], pControl, nGates);
    CopySharedBytesToGlobal(&seq.m_cuTarget[outputIndex], pTarget, nGates);
    CopySharedBytesToGlobal(&seq.m_cuGates[outputIndex], pGates, nGates);
  }
  seq.m_cuNumGates[threadIdx.x] = nGates;

  //printf("block: nGates Index: %d [%d]\n", blockIdx.x, seq.m_cuGates[blockIdx.x]);
}


void SynthesizeKernel(CudaSequence *pcuSeq, int nSequences)
{
  // Constants are scoped to a file, and cannot use extern..
  CS( cudaMemcpyToSymbol(gcuBitMask, gBitMask, sizeof(gBitMask)) );
  CS( cudaMemcpyToSymbol(gcuTernaryOps, gTernaryOps, sizeof(gTernaryOps)) );
  CS( cudaMemcpyToSymbol(gcuOpMap, gOpMap, sizeof(gOpMap)) );
  cuSynthesizeKernel<<<1, nSequences>>>(pcuSeq);
}

__device__ int Propagate(int outTerm, PINT pControl, PBYTE pTarget, PBYTE pOperation, int nGates)
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

__device__ void Process(int inTerm, int outTerm, int nBits, PINT pnGates, PINT pControl, PBYTE pTarget,  PBYTE pOperation)
{
  //printf("\n****** In,out:[%d, %d], nGates: [%d] ", inTerm, outTerm, *pnGates);
  outTerm = Propagate(outTerm, pControl, pTarget, pOperation, *pnGates);

  //printf("After Propgate: %d \n", outTerm);

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
       // printf("dir(%d) [C,T,O] [%d, %d, %d] ", dir, pControl[*pnGates], i, pOperation[*pnGates]);
        (*pnGates)++;
        outTerm = (~gcuBitMask[i] & outTerm) | (gcuBitMask[i] & inTerm);
       // printf(" => %d \n", outTerm);
      }
    }
  }
}