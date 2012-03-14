#pragma once
#include "ConductorCore.h"
#include "Generators/GeneratorCore.h"
#include "Synthesizers/SynthesizerCore.h"
#include "Sequence.h"
using namespace System;
using namespace System::IO;

namespace Conductor {
  class GeneticAlgorithm : public Core, public Helper::Result {
  private:
    Generator::Core *m_pGenerator;
    Synthesizer::Core *m_pSynthesizer;
    Sequence **m_pSeq;
    int m_nPopulation;
    int m_nRuns;
    int m_nGenerations;
    int m_ParentTotalFitness;
    int m_BestFit;

  public:
    GeneticAlgorithm(int nBits, Generator::Core *pGen, Synthesizer::Core *pSyn) {
      m_pGenerator = pGen;
      m_pSynthesizer = pSyn;
      m_nRuns = 1;
      m_nGenerations = 1;
      m_nPopulation = 1;
      m_pSeq = new Sequence *[2*m_nPopulation]; // twice as many to hold children as well.
    }

    ~GeneticAlgorithm() {
      for(int i=0; i<2*m_nPopulation; i++)
        delete m_pSeq[i];

      delete m_pSeq;
    }

    void InitializePopulation()
    {
      for(int i=0; i<m_nPopulation; i++) {
        m_pSeq[i] = m_pGenerator->GetSequence();
        Helper::DumpSequence(m_pSeq[i]);
      }
    }

    void Process()
    {
      InitializePopulation();

      Utility::CStopWatch s;
      s.startTimer();
      m_BestFit = MAXLONG;

      for(int i=0; i<m_nRuns; i++)
        for(int g=0; g<m_nGenerations; g++)
          DoGeneration(g);

      s.stopTimer();
      PrintResult(1, s.getElapsedTime());
    }
      

    void DoGeneration(int gen)
    {
      m_ParentTotalFitness = 0;

      for(int i=0; i<m_nPopulation; i++)
        m_pSynthesizer->AddSequence(m_pSeq[i]);

      m_pSynthesizer->Process();

      for (int i=0; i<m_nPopulation; i++) {
        int qCost = m_pSeq[i]->QuantumCost();
        m_ParentTotalFitness += qCost;
        if ( (gen % 10) == 0)
          Console::Write("Gen: {0}, BestCost: {1}\r", gen, m_BestFit);

        if (m_BestFit > qCost) {
          m_BestFit = qCost;
          Console::WriteLine("Gen: {0}, BestCost: {1}", gen, m_BestFit);
          SaveResult(m_pSeq[i]);
        }
      }
    }
  };
}