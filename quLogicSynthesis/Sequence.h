#pragma once
class Sequence {
  int* m_pIn;
  int* m_pOut;
  int  m_nBits;
  int  m_nTerms;
  int  m_nQuantumCost;
  int* m_pControl;
  int* m_pTarget;
  int* m_pGate;
};