#pragma once
#include "ConductorCore.h"
#include "Generators/GeneratorCore.h"
#include "Synthesizers/SynthesizerCore.h"
#include "Sequence.h"
#include "Synthesizers/Ternary/Cuda/CudaSequence.h"
#include "GeneticAlgorithmParameters.h"
#include "Utilities/Rand.h"
using namespace System;
using namespace System::IO;

namespace Conductor {
  class GeneticAlgorithm : public Core, public Helper::Result, public GeneticAlgorithmParameters {
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
      m_nPopulation = NUMBER_OF_CUDA_BLOCKS;
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
          Console::Write("Gen: {0}, BestCost: {1}\n", gen, m_BestFit);

        if (m_BestFit > qCost) {
          m_BestFit = qCost;
          Console::WriteLine("Gen: {0}, BestCost: {1}\n", gen, m_BestFit);
          SaveResult(m_pSeq[i]);
        }
      }

      Breed();
      Cull();
    }
    // <outputs>
    void Cull()
    {
      for (int i=0; i<m_nPopulation; i++) {
        delete m_pSeq[i];
        m_pSeq[i] = m_pSeq[i+m_nPopulation];
      }
    }

    void Breed()
    {
      for (int i=0; i<m_nPopulation; i++) {
        Sequence *p1 = Roulette();  
        Sequence *p2 = Roulette();
        m_pSeq[i+m_nPopulation] = m_nCrossOver == 0 ? m_pGenerator->SinglePointCrossOver(p1, p2, m_Pc) : m_pGenerator->TwoPointCrossOver(p1, p2, m_Pc);
        m_pGenerator->Mutate(m_pSeq[i+m_nPopulation], m_Pm);
      }
    }

    Sequence *Roulette()
    {
      double rnd = Rand::Double();
      double val=0;

      for (int i=0; i < m_nPopulation; i++) {
        val += m_pSeq[i]->QuantumCost()/m_ParentTotalFitness;
        if (rnd < val)
          return m_pSeq[i];
      }

      return m_pSeq[m_nPopulation-1];
    }
  };
}