// quLogicSynthesis.cpp : main project file.

#include "stdafx.h"
#include "Winbase.h"
#include "Conductors/GeneticAlgorithm.h"
#include "Generators/Ternary/OrderedSet.h"
#include "Synthesizers/Ternary/Basic.h"
#include "Synthesizers/Ternary/Cuda/Basic.h"
#include "Utilities/Rand.h"

using namespace std;
using namespace System;

#define FILE_PATTERN "Ternary\\hwt"
int ternary()
{
  for (int nBits=5; nBits<=5; nBits++) {
    Config::SetRadix(3, nBits);

    int* pOut;
    Utility::FileSrc fs(nBits, FILE_PATTERN + Convert::ToString(nBits));

    while (pOut = fs.Next() ) {
      Helper::pOutput = pOut;
      Console::WriteLine("Function: " + fs.Name);
      Synthesizer::Core *pSyn = new Synthesizer::Ternary::Cuda::Basic(nBits);
      Generator::Core *pGen = new Generator::Ternary::OrderedSet(nBits, pOut);
      Conductor::Core *pAlgo = new Conductor::GeneticAlgorithm(nBits, pGen, pSyn);
      pAlgo->Process();
      delete pSyn;
      delete pGen;
      delete pAlgo;
    }
  }
  return 0;
}

int main()
{
  Rand::Initialize();

  ternary();
}