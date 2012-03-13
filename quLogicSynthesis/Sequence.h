#pragma once
namespace Helper {
  extern int* pOutput;
}

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

  void GenerateOutput(int* pOut)
  {
    // Arrange output minterms to match their input minterms according to original specification.
    m_pOut = new int[m_nTerms];
    for(int i=0; i<m_nTerms; i++) {
      m_pOut[i] = pOut[m_pIn[i]];
    }

#ifdef _DEBUG
    // Verify that the input/output sequence match
    for(int i=0; i<m_nTerms; i++) {
      if (m_pOut[i] != Helper::pOutput[m_pIn[i]])
        throw "They don't match. WHY?";
    }
#endif
  }
  ~Sequence()
  {
    delete m_pIn;
    delete m_pOut;
    delete m_pControl;
    delete m_pTarget;
    delete m_pGate;
  }
};