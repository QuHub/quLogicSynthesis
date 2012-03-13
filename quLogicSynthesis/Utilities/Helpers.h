#pragma once
#include "Sequence.h"
#include "Config.h"

namespace Helper {
  extern int* pOutput;
  void DumpSequence(int* pIn, int* pOut, int nCount);
  void DumpSequence(Sequence *pSeq);
  void DumpSequence(int* pOut, int nCount);
  int inline BitsToTerms(int nBits) {return (int)pow((float)Config::Radix(), nBits);}
}