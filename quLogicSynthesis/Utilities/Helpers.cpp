#include "stdafx.h"
#include "Helpers.h"
namespace Helper {
  int* pOutput;
  Utility::CStopWatch StopTimer;
#ifdef _DEBUG
  void DumpSequence(int* pIn, int* pOut, int nCount)
  {
      for (int i=0; i<nCount; i++)
        cout << pIn[i] << " - " << pOut[i] << endl;
  }

  void DumpSequence(Sequence *pSeq)
  {
      for (int i=0; i<pSeq->m_nTerms; i++)
        cout << pSeq->m_pIn[i] << " - " << pSeq->m_pOut[i] << endl;
  }

  void DumpSequence(int* pOut, int nCount)
  {
      for (int i=0; i<nCount; i++)
        cout << i << " - " << pOut[i] << endl;
  }
#else
    // TODO: Need to stub it out
    #define DumpSequences() {}
#endif

  int InRadixDigits(int term)
  {
    int result = 0;

    for(int i=0; i<8*sizeof(int)/Config::RadixBits(); i++) {
      int digit = (term >> i*Config::RadixBits()) & Config::RadixMask();
      result += digit * pow(10.0, i);
    }

    return result;
  }
}