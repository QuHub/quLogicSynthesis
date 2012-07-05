// quLogicSynthesis.cpp : main project file.

#include "stdafx.h"
#include "Winbase.h"
#include "Conductors/GeneticAlgorithm.h"
#include "Conductors/Ternary/PassThrough.h"
#include "Generators/Ternary/OrderedSet.h"
#include "Generators/Ternary/NaturalOrder.h"
#include "Synthesizers/Ternary/Basic.h"
#include "Synthesizers/Ternary/Cuda/Basic.h"
#include "Utilities/Rand.h"

using namespace std;
using namespace System;

#define FILE_PATTERN "Ternary\\hwt"
int GATernary();
int TestTernary();

int main()
{
  GATernary();
  TestTernary();
}

int TestTernary()
{
  for (int nBits=2; nBits<=4; nBits++) {
    Config::SetRadix(3, nBits);

    int* pOut;
    Utility::FileSrc fs(nBits, FILE_PATTERN + Convert::ToString(nBits));

    while (pOut = fs.Next() ) {
      Console::WriteLine("Processing Next Sequence");
      Helper::pOutput = pOut;
      Console::WriteLine("Function: " + fs.Name);
      Synthesizer::Core *pSyn = new Synthesizer::Ternary::Cuda::Basic(nBits);
      Generator::Core *pGen = new Generator::Ternary::NaturalOrder(nBits, pOut);
      Conductor::Core *pAlgo = new Conductor::PassThrough(nBits, pGen, pSyn);
      pAlgo->Process();
      delete pSyn;
      delete pGen;
      delete pAlgo;
    }
  }
  return 0;
}

int GATernary()
{
  Rand::Initialize();
  for (int nBits=3; nBits<=6; nBits++) {
    Config::SetRadix(3, nBits);

    int* pOut;
    Utility::FileSrc fs(nBits, FILE_PATTERN + Convert::ToString(nBits));

    while (pOut = fs.Next() ) {
      Console::WriteLine("Processing Next Sequence");
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

