#pragma once
#include "ConductorCore.h"
#include "Generators/GeneratorCore.h"
#include "Synthesizers/SynthesizerCore.h"
#include "Sequence.h"

namespace Conductor {
  class GeneticAlgorithm : public Core {
  private:
    Generator::Core *m_pGenerator;
    Synthesizer::Core *m_pSynthesizer;
    Sequence **m_pSeq;
    int m_nPopulation;
    int m_nRuns;
    int m_nGenerations;
    int m_ParentTotalFitness;

  public:
    GeneticAlgorithm(int nBits, Generator::Core *pGen, Synthesizer::Core *pSyn) {
      m_pGenerator = pGen;
      m_pSynthesizer = pSyn;
      m_nRuns = 20;
      m_nGenerations = 10;
      m_nPopulation = 100;
      m_pSeq = new Sequence *[2*m_nPopulation]; // twice as many to hold children as well.
    }

    ~GeneticAlgorithm() {
      for(int i=0; i<2*m_nPopulation; i++)
        delete m_pSeq[i];

      delete m_pSeq;
    }

    void InitializePopulation()
    {
      for(int i=0; i<m_nPopulation; i++)
        m_pSeq[i] = m_pGenerator->GetSequence();
    }

    void Process()
    {
      InitializePopulation();

      Utilities::CStopWatch s;
      s.startTimer();

      for(int i=0; i<m_nRuns; i++)
        for(int g=0; g<m_nGenerations; g++)
          DoGeneation(g);

      s.stopTimer();
    }
      

    void DoGeneration(int g)
    {
      m_ParentTotalFitness = 0;

      for(int i=0; i<m_nPopulation; i++)
        m_pSynthesizer->AddSequence(m_pSeq[i]);

      m_pSynthesizer->Process();
      //m_pSynthesizer->GetResults();
    }
  };
}