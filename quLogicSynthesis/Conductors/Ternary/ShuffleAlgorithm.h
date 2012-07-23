#pragma once
#include "ConductorCore.h"
#include "../../Generators/GeneratorCore.h"
#include "../../Synthesizers/SynthesizerCore.h"
#include "../../Sequence.h"
#include "../../Synthesizers/Ternary/Cuda/CudaSequence.h"
#include "../../Utilities/Rand.h"
#include "../../Utilities/Thread.h"

using namespace System;
using namespace System::IO;

#define NUM_CUDA_BLOCKS 512
#define NUM_RUNS 20

namespace Conductor {
  class Shuffle : public Core, public Helper::Result {
  public:
    Generator::Core *m_pGenerator;
    Synthesizer::Core *m_pSynthesizer;
    Sequence **m_pSeq;
    int m_nSequences;

  public:
    Shuffle(int nBits, Generator::Core *pGen, Synthesizer::Core *pSyn) {
      m_pGenerator = pGen;
      m_pSynthesizer = pSyn;
      m_nSequences = 2 * NUM_CUDA_BLOCKS;
    }

    ~Shuffle() {
    }

    void Process()
    {
      int bestCost = MAXINT;
      Sequence *pSeq;

      Helper::StopTimer.Start();
      Utility::CStopWatch s;
      s.Start();
      m_pSynthesizer->Initialize();
      for (int j=0; j<512; j++) {
        Console::Write("Batch: {0}\r", j);
        m_pGenerator->ReleaseSequences();
        m_pSynthesizer->m_Sequences.clear();
        for (int i=0; i<1024; i++) {
          m_pSynthesizer->AddSequence(m_pGenerator->GetSequence());
        }
        m_pSynthesizer->Process();
        m_pSynthesizer->PostProcess();

        for(int i=0; i<1024; i++) {
          int nGates = m_pSynthesizer->m_Sequences[i]->m_nGates;
          if (bestCost > nGates) {
              bestCost = nGates;
              SaveResult(m_pSynthesizer->m_Sequences[i]);
          }
        }
      }
      s.Stop();
      PrintResult(0, "PassThrough", s.getElapsedTime());
    }
  };
}