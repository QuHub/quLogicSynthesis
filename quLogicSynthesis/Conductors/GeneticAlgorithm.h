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


#define ROLETTE_SIZE (sizeof(m_RoletteWheel)/sizeof(m_RoletteWheel[0]))

namespace Conductor {
  class GeneticAlgorithm : public Core, public Helper::Result, public GeneticAlgorithmParameters {
  private:
    Generator::Core *m_pGenerator;
    Synthesizer::Core *m_pSynthesizer;
    Sequence **m_pSeq;
    int m_nPopulation;
    int m_ParentTotalFitness;
    int m_BestFit;
    bool m_fReport;
    int m_RoletteWheel[1000];

  public:
    GeneticAlgorithm(int nBits, Generator::Core *pGen, Synthesizer::Core *pSyn) {
      m_pGenerator = pGen;
      m_pSynthesizer = pSyn;
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

      Helper::StopTimer.Start();
      while(NextGeneticAlgorithmParameters()) {
          Utility::CStopWatch s;
          s.Start();
          m_BestFit = MAXLONG;

          for(int i=0; i<m_nRuns; i++)
            for(int g=0; g<m_nGenerations; g++)
              DoGeneration(g);

          s.Stop();
          //PrintResult(1, s.getElapsedTime());
          P(String::Format("NextGeneticAlgorithmParameter: {0}\n", Helper::StopTimer.getElapsedTime()));
      }
    }
    void DoGeneration(int gen)
    {
      P(String::Format("\nProcessing Generation: {0}\n", gen));
      m_ParentTotalFitness = 0;

      m_fReport = (gen % 10) == 0;
      m_pSynthesizer->Initialize();

      P(String::Format("Initialize: {0}\n", Helper::StopTimer.getElapsedTime()));

      for(int i=0; i<m_nPopulation; i++)
        m_pSynthesizer->AddSequence(m_pSeq[i]);

      P(String::Format("AddSequence: {0}\n", Helper::StopTimer.getElapsedTime()));
      m_pSynthesizer->Process();

      P(String::Format("Process: {0}\n", Helper::StopTimer.getElapsedTime()));

      for (int i=0; i<m_nPopulation; i++) {
        int qCost = m_pSeq[i]->QuantumCost();
        m_ParentTotalFitness += qCost;
        if ( m_fReport ) {
          Console::Write("Gen: {0}, BestCost: {1}\n", gen, m_BestFit);
          m_fReport = false;
        }

        if (m_BestFit > qCost) {
          m_BestFit = qCost;
          Console::WriteLine("Gen: {0}, BestCost: {1}\n", gen, m_BestFit);
          //SaveResult(m_pSeq[i]);
        }
      }

      P(String::Format("QuantumCost: {0}\n", Helper::StopTimer.getElapsedTime()));
      Breed();
      P(String::Format("Breed: {0}\n", Helper::StopTimer.getElapsedTime()));
      Cull();
      P(String::Format("Cull: {0}\n", Helper::StopTimer.getElapsedTime()));
    }
    // <outputs>
    void Cull()
    {
      for (int i=0; i<m_nPopulation; i++) {
        delete m_pSeq[i];
        m_pSeq[i] = m_pSeq[i+m_nPopulation];
      }
    }

    void InitializeRoletteWheel()
    {
      double scale = (double)ROLETTE_SIZE/(double)m_ParentTotalFitness;
      int total = 0;

      int index = 0;
      for (int i=0; i<m_nPopulation; i++) {
        total += m_pSeq[i]->QuantumCost();
        int last = Math::Round(total * scale);
        while(index < last && index < ROLETTE_SIZE)
          m_RoletteWheel[index++] = i;

        index = last;
      }
    }

    void Breed()
    {
      InitializeRoletteWheel();
      for (int i=0; i<m_nPopulation; i++) {
        Sequence *p1 = Roulette();  
        Sequence *p2 = Roulette();
        m_pSeq[i+m_nPopulation] = m_nCrossOver == 0 ? m_pGenerator->SinglePointCrossOver(p1, p2, m_Pc) : m_pGenerator->TwoPointCrossOver(p1, p2, m_Pc);
        m_pGenerator->Mutate(m_pSeq[i+m_nPopulation], m_Pm);
      }
    }

    Sequence *Roulette()
    {
      int rnd = Rand::Integer(ROLETTE_SIZE - 1);
      return m_pSeq[m_RoletteWheel[rnd]];
    }
  };
}