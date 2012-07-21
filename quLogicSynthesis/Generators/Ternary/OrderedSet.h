#pragma once
#include "../GeneratorCore.h"
#include "Generators/Ternary/Hasse.h"
#include "Utilities/Rand.h"

namespace Generator {
  namespace Ternary {

    Utility::Ternary::Hasse *m_pHasse;
    class OrderedSet: public Generator::Core{
    public:
      int *m_pOut;
      int m_nBits;
      int *m_BandBoundary;
      inline int nBands() {return m_pHasse->m_nBands;}
      inline int nTerms() {return m_pHasse->m_nTerms;}
      vector<Sequence*> m_pSequences;

      OrderedSet(int nBits, int * pOut)
      {
        m_pOut = pOut;
        m_nBits = nBits;
        m_pHasse = new Utility::Ternary::Hasse(nBits);

        // Allocate array of band boundaries for crossover operations.
        m_BandBoundary = new int[nBands()];

        m_BandBoundary[0] = (int)m_pHasse->m_pBands[0].size();
        for (int j=1; j < nBands(); j++) {
          m_BandBoundary[j] = m_BandBoundary[j-1] + (int)m_pHasse->m_pBands[j].size();
        }
      }

      ~OrderedSet() 
      {
        delete m_BandBoundary;
        delete m_pHasse;
        m_pSequences.clear();
      }

      virtual Sequence *GetSequence()
      {
        Sequence *pSeq = new Sequence();
        
        pSeq->m_nBits = m_nBits;
        pSeq->m_nTerms = nTerms();
        pSeq->m_pIn = m_pHasse->GetSequence();   
        pSeq->GenerateOutput(m_pOut);

        m_pSequences.push_back(pSeq);

        return pSeq;
      }

      void ReleaseSequences()
      {
        for(int i=0; i<m_pSequences.size(); i++)
          delete m_pSequences[i];
        m_pSequences.clear();

        Console::WriteLine("Number of sequences{0}: ", m_pSequences.size());
      }

      Sequence* SinglePointCrossOver(Sequence *p1, Sequence *p2, double prob)
      {
        int cost1 = p1->QuantumCost();
        int cost2 = p2->QuantumCost();

        Sequence *p = new Sequence(cost1 < cost2 ? *p1 : *p2);  // Best Fit

        if (Rand::Double() < prob) {
          Sequence *q = cost1 < cost2 ? p2 : p1;                // Less Fit

          int nFirst = m_BandBoundary[Rand::Integer(nBands())];

          CopyMemory(p->m_pIn + nFirst, q->m_pIn + nFirst, (nTerms() - nFirst) * sizeof(int));
        }
        p->GenerateOutput(m_pOut);
        return p;
      }

      Sequence* TwoPointCrossOver(Sequence *p1, Sequence *p2, double prob)
      {
        int cost1 = p1->QuantumCost();
        int cost2 = p2->QuantumCost();

        Sequence *p = new Sequence(cost1 < cost2 ? *p1 : *p2);  // Best Fit

        if (Rand::Double() < prob) {
          Sequence *q = cost1 < cost2 ? p2 : p1;                // Less Fit

          int nFirst = m_BandBoundary[Rand::Integer(nBands())];
          int nSecond = m_BandBoundary[Rand::Integer(nBands())];

          if(nFirst > nSecond) {
            int tmp = nSecond;
            nSecond = nFirst;
            nFirst = tmp;
          }

          CopyMemory(p->m_pIn + nFirst, q->m_pIn + nFirst, (nSecond - nFirst) * sizeof(int));
        }
        p->GenerateOutput(m_pOut);
        return p;
      }

      void Mutate(Sequence *p1, double prob)
      {
        if (Rand::Double() > prob) return;

        int band = Rand::Integer(nBands()-1);
        int nStart= m_BandBoundary[band];
        int nEnd = m_BandBoundary[band+1];
        int nFirst = nStart + Rand::Integer(nEnd - nStart);
        int nSecond = nStart + Rand::Integer(nEnd - nStart);

        // Mutate through swap..
        int tmp = p1->m_pIn[nFirst];
        p1->m_pIn[nFirst] = p1->m_pIn[nSecond];
        p1->m_pIn[nSecond] = tmp;
      }
    };
  }
}


