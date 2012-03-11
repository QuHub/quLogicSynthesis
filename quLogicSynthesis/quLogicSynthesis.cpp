// quLogicSynthesis.cpp : main project file.

#include "stdafx.h"
#include "Conductors/GeneticAlgorithm.h"
#include "Generators/Ternary/OrderedSet.h"
#include "Synthesizers/Ternary/Basic.h"

using namespace System;

int main(array<System::String ^> ^args)
{
  Console::WriteLine(L"Hello World");
  int nBits = 2;

//  FileSrc fs(nBits, FILE_PATTERN + Convert::ToString(2));

  int * pOut;
  Synthesizer::Core *pSyn = new Synthesizer::Ternary::Basic();
  Generator::Core *pGen = new Generator::Ternary::OrderedSet(nBits, pOut);
  Conductor::Core *pAlgo = new Conductor::GeneticAlgorithm(nBits, pGen, pSyn);
  return 0;
}
