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
    Sequence **m_pSeq;
    int m_nSequences;

  public:
    Shuffle(int nBits, Generator::Core *pGen, Synthesizer::Core *pSyn) {
      m_pGenerator = pGen;
      m_pSynthesizer = pSyn;
      m_nSequences = 2 * NUM_CUDA_BLOCKS;
      m_pSeq = new Sequence *[m_nSequences];  // two devices
    }

    ~Shuffle() {
      for(int i=0; i<sizeof(m_pSeq)/sizeof(m_pSeq[0]); i++)
        delete m_pSeq[i];

      delete m_pSeq;
    }

    void InitializePopulation()
    {
      for(int i=0; i<m_nSequences; i++) {
        m_pSeq[i] = m_pGenerator->GetSequence();
      }
    }

    void Process()
    {
      InitializePopulation();

      //m_pSynthesizer->Initialize();
      //m_pSynthesizer->AddSequence(m_pSeq);
      //m_pSynthesizer->Process();

      //SaveResult(m_pSeq);
    }
  };
}