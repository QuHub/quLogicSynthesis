#pragma once
namespace Synthesizer {
  class Core {
  public:
    vector<Sequence*> m_Sequences;
    int m_nBits;

  public:
    Core() {}
    Core(int nBits) { m_nBits = nBits; }
    ~Core(){
      for (int i=0; i<m_Sequences.size(); i++)
        delete m_Sequences[i];
    }
    void AddSequence(Sequence *pSeq){ m_Sequences.push_back(pSeq); }
    void Initialize() { m_Sequences.clear(); }

    virtual void Process(){
#ifdef _DEBUG
      for(int i=0; i<m_Sequences.size(); i++) {
        Helper::DumpSequence(m_Sequences[i]->m_pIn, m_Sequences[i]->m_pOut, m_Sequences[i]->m_nTerms);
      }
#endif
    }
  };
}

