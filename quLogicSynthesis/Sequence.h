#pragma once
namespace Helper {
  extern int* pOutput;
}

#include "Config.h"
#include <windows.h>
class Sequence {
public:
  int* m_pIn;
  int* m_pOut;
  int  m_nBits;
  int  m_nTerms;
  int  m_nGates;
  int* m_pControl;
  byte* m_pTarget;
  byte* m_pGates;
  int* m_pInputRadixBuffer;
  int* m_pOutputRadixBuffer;

  Sequence(){Init();}

  Sequence(const Sequence& base) {
    Init();
    m_nBits = base.m_nBits;
    m_nTerms = base.m_nTerms;
    m_nGates = base.m_nGates;
    m_pIn = new int[m_nTerms];
//    m_pOut = new int[m_nTerms];
    // Again, we don't need to copy these on the Copy Constructor...
//    CopyMemory(m_pControl, base.m_pControl, by(MAX_GATES));
//    CopyMemory(m_pTarget, base.m_pTarget, MAX_GATES);
//    CopyMemory(m_pGates, base.m_pGates, MAX_GATES);
    CopyMemory(m_pIn, base.m_pIn, m_nTerms * sizeof(int));
//    CopyMemory(m_pOut, base.m_pOut, m_nTerms * sizeof(int));
  }

  void Init() {
    m_pInputRadixBuffer = m_pOutputRadixBuffer = NULL;
    m_pControl = (LPINT)VirtualAlloc(NULL,by(MAX_GATES), MEM_COMMIT, PAGE_READWRITE);
    m_pGates = (LPBYTE)VirtualAlloc(NULL,MAX_GATES, MEM_COMMIT, PAGE_READWRITE);
    m_pTarget = (LPBYTE)VirtualAlloc(NULL,MAX_GATES, MEM_COMMIT, PAGE_READWRITE);
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
    VirtualFree(m_pControl, 0,MEM_RELEASE); 
    VirtualFree(m_pTarget, 0,MEM_RELEASE); 
    VirtualFree(m_pGates, 0,MEM_RELEASE); 
    if (m_pInputRadixBuffer != NULL)
      delete m_pInputRadixBuffer;
    if (m_pOutputRadixBuffer != NULL)
      delete m_pOutputRadixBuffer;
  }
};