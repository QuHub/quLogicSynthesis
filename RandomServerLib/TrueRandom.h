#pragma once
#include "windows.h"
#include "Thread.h"
namespace RandomServer
{
  public class  TrueRandom : public CThread
  {
  protected:
    List<double> m_numbers;
  public:
    TrueRandom(void);

    void Fill();
    virtual DWORD Run(LPVOID args);
  };
}