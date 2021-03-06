#pragma once
#include "qrbg-cpp.h"
#include "math.h"

#define BUF_SIZE (sizeof(m_buffer)/sizeof(m_buffer[0]))
namespace Rand
{
  QRBG* m_rndService=NULL;
  double m_buffer[1024*1024];
  int m_index=BUF_SIZE;

  void Init(void)
  {
    if (m_rndService)
      return;

    // Create random service object
    try {
      m_rndService = new QRBG;
    } catch (QRBG::NetworkSubsystemError) {
      printf("Network error!");
      return ;
    } catch (...) {
      printf("Failed to create QRBG client object!");
      return ;
    }

    // Login to server
    try {
      m_rndService->defineServer("random.irb.hr", 1227);
      m_rndService->defineUser("mhawash", "seven11");
    } catch (QRBG::InvalidArgumentError e) {
      printf("Invalid hostname/port, or username/password specified! \n");
      delete m_rndService;
      return ;
    }
  }

  void TearDown(void)
  {
    delete m_rndService;
  }


  void Fill()
  {
    Init();
    if (m_index == BUF_SIZE) {
      m_rndService->getDoubles(m_buffer, BUF_SIZE);
      m_index = 0;
    }
  }

  double Current() {return m_buffer[m_index];}
  int IInteger(int nRange=MAX_INT) {return (int)(m_buffer[m_index] * m_nRange);}
  double Prev() {return m_buffer[m_index-1];}
  double Double()  { Fill(); return m_buffer[m_index++];}
  unsigned long Bit()      { Fill(); return (m_buffer[m_index++] < 0.5 ? 0 : 1);}
  bool Truth()    { Fill(); return (m_buffer[m_index++] < 0.5 ? false : true);}

};

