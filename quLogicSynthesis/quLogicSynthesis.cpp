// quLogicSynthesis.cpp : main project file.

#include "stdafx.h"
#include "Conductors/GeneticAlgorithm.h"
#include "Generators/Ternary/OrderedSet.h"
#include "Synthesizers/Ternary/Basic.h"
#include "Synthesizers/Ternary/Cuda/Basic.h"

using namespace std;
using namespace System;

int main(array<System::String ^> ^args)
{
  Console::WriteLine(L"Hello World");
  int nBits = 3;

//  FileSrc fs(nBits, FILE_PATTERN + Convert::ToString(2));

  Config::SetRadix(3, nBits);
  int pOut[] = {1,2,0,3,5,6,4,8,7,10,9,13,12,11,15,14,20,19,17,16,18,25,26,22,21,23,24};
  Helper::pOutput = pOut;

  Helper::DumpSequence(pOut, 27);

  Synthesizer::Core *pSyn = new Synthesizer::Ternary::Cuda::Basic(nBits);
  Generator::Core *pGen = new Generator::Ternary::OrderedSet(nBits, pOut);
  Conductor::Core *pAlgo = new Conductor::GeneticAlgorithm(nBits, pGen, pSyn);
  pAlgo->Process();
  return 0;
}
