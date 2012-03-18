#pragma once
#include <Windows.h>
#define NUMBER_OF_CUDA_BLOCKS 512 
typedef struct  
{
  PINT m_pIn, m_pOut, m_pTarget, m_pControl, m_pOperation;
  PINT m_cuIn, m_cuOut, m_cuControl; 
  PINT  m_cuTarget, m_cuOperation;
  PINT m_pnGates, m_cuGates;
  PINT m_pgBitMask;
  PINT m_pgOpMapi[3], m_pgTernaryOps[3];
  int m_nTerms, m_nBits;
  int m_nMaxGates;
  int m_nSequences;
  int m_nVectorSizeBytes;
} CudaSequence;