#pragma once
#include "math.h"
#include "windows.h"

namespace Rand
{
  extern double m_buffer[1024*1024];
  extern int m_index;

  void Init(void);
  void TearDown(void);
  void Fill();
  inline double Current() {return m_buffer[m_index];}
  inline double Prev() {return m_buffer[m_index-1];}
  inline int  Integer(int nRange= INT_MAX) {return (int)(m_buffer[m_index-1] * nRange);}
  inline double Double()  { Fill(); return m_buffer[m_index++];}
  inline unsigned long Bit()      { Fill(); return (m_buffer[m_index++] < 0.5 ? 0 : 1);}
  inline bool Truth()    { Fill(); return (m_buffer[m_index++] < 0.5 ? false : true);}
};

