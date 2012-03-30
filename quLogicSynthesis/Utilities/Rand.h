#pragma once
#include "windows.h"
#include "Thread.h"

namespace Rand
{
  public class Rand : public CThread
  {
  protected:
    double m_numbers[10000];
    int m_reader;
  public:
    Rand(void);

    void Fill();
    virtual DWORD Run(LPVOID args);
    void Rand::Extract(String^ line, int index);
    double Double(); 
  };

  extern Rand *m_pRandom;

  inline void Initialize() { m_pRandom = new Rand();}
  inline double Double() {return m_pRandom->Double();}
  inline int Integer(int range=INT_MAX) {return (int)(range*m_pRandom->Double());}
}