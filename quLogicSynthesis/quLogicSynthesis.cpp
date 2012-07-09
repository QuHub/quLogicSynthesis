// quLogicSynthesis.cpp : main project file.

#include "stdafx.h"
#include "Winbase.h"
#include "Conductors/Ternary/GeneticAlgorithm.h"
#include "Conductors/Ternary/PassThrough.h"
#include "Conductors/Ternary/ShuffleAlgorithm.h"
#include "Generators/Ternary/OrderedSet.h"
#include "Synthesizers/Ternary/Basic.h"
#include "Synthesizers/Ternary/Cuda/Basic.h"
#include "Utilities/Rand.h"

using namespace std;
using namespace System;

#define FILE_PATTERN "Ternary\\hwt"
int GATernary();
int PassThrough();
int RandomAlgorithm();

int main()
{
  //RandomAlgorithm();
  PassThrough();
//  GATernary();
}

int PassThrough()
{
  for (int nBits=5; nBits<=6; nBits++) {
    Config::SetRadix(3, nBits);

    int* pOut;
    Utility::FileSrc fs(nBits, FILE_PATTERN + Convert::ToString(nBits));

    while (pOut = fs.Next() ) {
      Console::WriteLine("Processing Next Sequence");
      Helper::pOutput = pOut;
      Console::WriteLine("Function: " + fs.Name);
      Synthesizer::Core *pSyn = new Synthesizer::Ternary::Cuda::Basic(nBits);
      Generator::Core *pGen = new Generator::Ternary::OrderedSet(nBits, pOut);
      Conductor::Core *pAlgo = new Conductor::PassThrough(nBits, pGen, pSyn);
      pAlgo->Process();
      delete pSyn;
      delete pGen;
      delete pAlgo;
    }
  }
  return 0;
}

int RandomAlgorithm()
{
  Rand::Initialize();
  for (int nBits=6; nBits<=6; nBits++) {
    Config::SetRadix(3, nBits);

    int* pOut;
    Utility::FileSrc fs(nBits, FILE_PATTERN + Convert::ToString(nBits));

    while (pOut = fs.Next() ) {
      Console::WriteLine("Processing Next Sequence");
      Helper::pOutput = pOut;
      Console::WriteLine("Function: " + fs.Name);
      Synthesizer::Core *pSyn = new Synthesizer::Ternary::Cuda::Basic(nBits);
      Generator::Core *pGen = new Generator::Ternary::OrderedSet(nBits, pOut);
      Conductor::Core *pAlgo = new Conductor::Shuffle(nBits, pGen, pSyn);
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
  for (int nBits=5; nBits<=6; nBits++) {
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

