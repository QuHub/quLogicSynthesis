#pragma once
#include <Windows.h>
typedef struct  
{
  PINT m_pIn, m_pOut, m_pControl;
  PINT m_cuIn, m_cuOut, m_cuControl; 

  LPBYTE m_pTarget, m_pGates;
  LPBYTE  m_cuTarget, m_cuGates;

  PINT m_pnGates, m_cuNumGates;
  PINT m_pgBitMask;
  PINT m_pgOpMapi[3], m_pgTernaryOps[3];
  int m_nTerms, m_nBits;
  int m_nMaxGates;
  int m_nSequences;
  int m_nVectorSizeBytes;
} CudaSequence;