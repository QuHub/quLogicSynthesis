#pragma once
#include "ConductorCore.h"
#include "Generators/GeneratorCore.h"
#include "Synthesizers/SynthesizerCore.h"
#include "Sequence.h"
#include "Synthesizers/Ternary/Cuda/CudaSequence.h"
#include "Utilities/Rand.h"
using namespace System;
using namespace System::IO;


#define ROLETTE_SIZE (sizeof(m_RoletteWheel)/sizeof(m_RoletteWheel[0]))

namespace Conductor {
  class PassThrough : public Core, public Helper::Result {
  private:
    Generator::Core *m_pGenerator;
    Synthesizer::Core *m_pSynthesizer;
    Sequence *m_pSeq;

  public:
    PassThrough(int nBits, Generator::Core *pGen, Synthesizer::Core *pSyn) {
      m_pGenerator = pGen;
      m_pSynthesizer = pSyn;
    }

    ~PassThrough() {
      delete m_pSeq;
    }

    void Process()
    {
      m_pSeq = m_pGenerator->GetSequence();

      Helper::StopTimer.Start();
      Utility::CStopWatch s;
      s.Start();
      m_pSynthesizer->Initialize();
      for (int i=0; i<1024; i++) {
          m_pSynthesizer->AddSequence(m_pSeq);
      }
      m_pSynthesizer->Process();

      s.Stop();
      SaveResult(m_pSeq);
      PrintResult(0, "PassThrough", s.getElapsedTime());
    }
  };
}