#pragma once
#include "ConductorCore.h"
#include "../../Generators/GeneratorCore.h"
#include "../../Synthesizers/SynthesizerCore.h"
#include "../../Sequence.h"
#include "../../Synthesizers/Ternary/Cuda/CudaSequence.h"
#include "../../Utilities/Rand.h"
#include "../../Utilities/Thread.h"
#include "CudaCore.h"

using namespace System;
using namespace System::IO;

#define NUM_CUDA_BLOCKS 512
#define NUM_RUNS 20

namespace Conductor {
  class Shuffle : public Core, public Helper::Result {
  public:
    Generator::Core *m_pGenerator;
    Synthesizer::Core *m_pSynthesizer;
    Sequence *m_pSeq;
    HANDLE *m_phMutex;

  public:
    Shuffle(int nBits, Generator::Core *pGen, Synthesizer::Core *pSyn) {
      m_pGenerator = pGen;
      m_pSynthesizer = pSyn;
    }

    ~Shuffle() {
      delete m_pSeq;
    }

    void Process()
    {
      // Launch Factory for generating sequences.
      Factory factory1, factory2;
      factory1.Start(this);
      Sleep(1000);
      //factory2.Start(this);

      // Launch two threads (one for each core)
      CudaCore core1, core2;

      core1.Start(&factory1);

      for(int i=0; i<NUM_RUNS; i++) {
        Sleep(10000);
        //core2.Start(&factory2);
      }

      Helper::StopTimer.Start();
      Utility::CStopWatch s;
      s.Start();
      m_pSynthesizer->Initialize();
      m_pSynthesizer->AddSequence(m_pSeq);
      m_pSynthesizer->Process();

      s.Stop();
      SaveResult(m_pSeq);
      PrintResult(0, "Shuffle", s.getElapsedTime());
    }
  };
}