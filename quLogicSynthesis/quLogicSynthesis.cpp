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
  int nBits = 2;

//  FileSrc fs(nBits, FILE_PATTERN + Convert::ToString(2));

  Config::SetRadix(3);
  int pOut[] = {1,2,0,3,5,6,4,8,7};
  Helper::pOutput = pOut;

  Helper::DumpSequence(pOut, 9);

  Synthesizer::Core *pSyn = new Synthesizer::Ternary::Cuda::Basic(nBits);
  Generator::Core *pGen = new Generator::Ternary::OrderedSet(nBits, pOut);
  Conductor::Core *pAlgo = new Conductor::GeneticAlgorithm(nBits, pGen, pSyn);
  pAlgo->Process();
  return 0;
}
