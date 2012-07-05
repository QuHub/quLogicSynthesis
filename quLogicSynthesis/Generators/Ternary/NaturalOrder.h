#pragma once
#include "../GeneratorCore.h"
#include "Utilities/Rand.h"

namespace Generator {
  namespace Ternary {

    class NaturalOrder: public Generator::Core{
    public:
      int *m_pOut;
      int m_nBits;
      int *m_BandBoundary;
      inline int nBands() {return m_pHasse->m_nBands;}
      inline int nTerms() {return (int)pow(Config::Radix(),(double)m_nBits);;}
      vector<Sequence*> m_pSequences;

      NaturalOrder(int nBits, int * pOut)
      {
        m_pOut = pOut;
        m_nBits = nBits;
      }

      ~NaturalOrder()
      {
        m_pSequences.clear();
      }

      virtual Sequence *GetSequence()
      {        
        Sequence *pSeq = new Sequence();
        
        pSeq->m_nBits = m_nBits;
        pSeq->m_nTerms = nTerms();
        pSeq->m_pIn =  new int[nTerms()];

        for (int i=0; i<nTerms(); i++) {
          pSeq->m_pIn[i] = i;
        }

        pSeq->GenerateOutput(m_pOut);
        m_pSequences.push_back(pSeq);

        return pSeq;
      }
    };
  }
}


