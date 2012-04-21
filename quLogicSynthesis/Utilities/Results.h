#pragma once
#pragma managed
#include <fstream>

#define RECYCLE(x) if (x) {delete x; x=NULL;}
#define COPY(dst, src, n) {if(dst) delete dst; dst = new int[n]; CopyMemory(dst, src, n);}
using namespace System::IO;
using namespace System;
using namespace std;

namespace Helper {
  class Result
  {
  public:
    Sequence *m_pSeq;
    Result(void) { m_pSeq = NULL; }

    void SaveResult(Sequence *seq)
    {
      if(m_pSeq) delete m_pSeq;

      m_pSeq = new Sequence(*seq);
    }

    void PrintResult(int iteration, double Time)
    {
      char szTmp[1024];

      Directory::CreateDirectory( String::Format("..\\SaveData\\{0}-bits\\{1}", m_pSeq->m_nBits, Config::Date));
      sprintf(szTmp, "../SaveData/%d-bits/%d/%d-iteration.qsy", m_pSeq->m_nBits, Config::Date, iteration);
      ofstream fs(szTmp);
      fs << "Bit Count: " << m_pSeq->m_nBits << endl;
      fs << "Quantum Cost: " << m_pSeq->QuantumCost() << endl;

      fs << "Input Sequence (decimal and in 2 bits per digit):" << endl;
      for(int i=0;i< m_pSeq->m_nTerms; i++)
        fs << m_pSeq->m_pIn[i] << ", ";
      fs << endl;

      int *p = m_pSeq->InputForRadix();
      for(int i=0;i< m_pSeq->m_nTerms; i++) {
        fs.fill('0');  fs.width(Config::Bits());
        fs << Helper::InRadixDigits(p[i]) << ", ";
      }
      fs << endl;

      fs << "Output Sequence (decimal and in 2 bits per digit):" << endl;
      for(int i=0;i< m_pSeq->m_nTerms; i++)
        fs << m_pSeq->m_pOut[i] << ", ";
      fs << endl;

      p = m_pSeq->OutputForRadix();
      for(int i=0;i< m_pSeq->m_nTerms; i++) {
        fs.fill('0');  fs.width(Config::Bits());
        fs << Helper::InRadixDigits(p[i]) << ", ";
      }
      fs << endl;

      char* gates[] = {"+1", "+2", "01", "02", "12"};
      fs << "Control | Target | Operation" << endl;
      for(int i=0;i<m_pSeq->m_nGates;i++) {
        fs.fill('0');
        fs.width(Config::Bits());
        fs << Helper::InRadixDigits(m_pSeq->m_pControl[i]) << " | ";
        fs.fill(' ');
        fs.width(5);
        fs << m_pSeq->m_pTarget[i] << " | ";
        fs.width(2);
      }
      fs.close();
    }
  };
}

