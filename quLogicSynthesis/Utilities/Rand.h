#pragma once
#include "windows.h"
#include "Thread.h"

namespace Rand
{
  public class Rand : public CThread
  {
  protected:
    queue<double> m_numbers;
    HANDLE m_hMutex;
    bool m_forceRefill;
  public:
    Rand(void);
    ~Rand();

    void Fill(int fill=1000);
    void ReFill();
    virtual DWORD Run(LPVOID args);
    void Rand::Extract(String^ line);

    void Lock() { WaitForSingleObject(m_hMutex, INFINITE); }
    void Release(){::ReleaseMutex(m_hMutex); }
    double Double(); 
  };

  extern Rand *m_pRandom;

  inline void Initialize() { m_pRandom = new Rand();}
  inline double Double() {return m_pRandom->Double();}
  inline int Integer(int range=INT_MAX) {return (int)(range*m_pRandom->Double());}
  inline void Refill() {m_pRandom->ReFill();}
}