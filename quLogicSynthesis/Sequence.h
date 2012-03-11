#pragma once
class Sequence {
public:
  int* m_pIn;
  int* m_pOut;
  int  m_nBits;
  int  m_nTerms;
  int  m_nQuantumCost;
  int* m_pControl;
  int* m_pTarget;
  int* m_pGate;

  ~Sequence()
  {
    delete m_pIn;
    delete m_pOut;
    delete m_pControl;
    delete m_pTarget;
    delete m_pGate;
  }
};