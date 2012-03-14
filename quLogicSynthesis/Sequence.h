#pragma once
namespace Helper {
  extern int* pOutput;
}

#include "Config.h"
#include <windows.h>
class Sequence {
public:
  static const int OutputBufferSize = 200*1024;
  int* m_pIn;
  int* m_pOut;
  int  m_nBits;
  int  m_nTerms;
  int  m_nGates;
  int* m_pControl;
  int* m_pTarget;
  int* m_pOperation;
  int* m_pInputRadixBuffer;
  int* m_pOutputRadixBuffer;

  Sequence(){Init();}

  Sequence(const Sequence& base) {
    Init();
    m_nBits = base.m_nBits;
    m_nTerms = base.m_nTerms;
    m_nGates = base.m_nGates;
    m_pIn = new int[m_nTerms];
    m_pOut = new int[m_nTerms];
    CopyMemory(m_pTarget, base.m_pTarget, OutputBufferSize);
    CopyMemory(m_pControl, base.m_pControl, OutputBufferSize);
    CopyMemory(m_pOperation, base.m_pOperation, OutputBufferSize);
    CopyMemory(m_pIn, base.m_pIn, m_nTerms * sizeof(int));
    CopyMemory(m_pOut, base.m_pOut, m_nTerms * sizeof(int));
  }

  void Init() {
    m_pInputRadixBuffer = m_pOutputRadixBuffer = NULL;
    m_pControl = new int[OutputBufferSize];
    m_pOperation = new int[OutputBufferSize];
    m_pTarget = new int[OutputBufferSize];
  }

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

  int* InputForRadix()
  {
    if(m_pInputRadixBuffer == NULL)
      m_pInputRadixBuffer = new int[m_nTerms];

    for(int i=0; i<m_nTerms; i++)
      m_pInputRadixBuffer[i] = Config::RadixDigits(m_pIn[i]);

    return m_pInputRadixBuffer;
  }

  int* OutputForRadix()
  {
    if(m_pOutputRadixBuffer == NULL)
      m_pOutputRadixBuffer = new int[m_nTerms];

    for(int i=0; i<m_nTerms; i++)
      m_pOutputRadixBuffer[i] = Config::RadixDigits(m_pOut[i]);

    return m_pOutputRadixBuffer;
  }

  int QuantumCost()
  {
    return m_nGates;
  }

  ~Sequence()
  {
    delete m_pIn;
    delete m_pOut;
    delete m_pControl;
    delete m_pTarget;
    delete m_pOperation;
    if (m_pInputRadixBuffer != NULL)
      delete m_pInputRadixBuffer;
    if (m_pOutputRadixBuffer != NULL)
      delete m_pOutputRadixBuffer;
  }
};