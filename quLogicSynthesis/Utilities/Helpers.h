#pragma once
#include "Sequence.h"
#include "Config.h"
#include "StopWatch.h"

using namespace System;
using namespace std;
namespace Helper {
  extern int* pOutput;
  
  extern Utility::CStopWatch StopTimer;
  void Initialize();
  void DumpSequence(int* pIn, int* pOut, int nCount);
  void DumpSequence(Sequence *pSeq);
  void DumpSequence(int* pOut, int nCount);
  int inline BitsToTerms(int nBits) {return (int)pow((float)Config::Radix(), nBits);}
  int InRadixDigits(int term);
  void inline Print(String^ x) {Console::Write(x);}
}