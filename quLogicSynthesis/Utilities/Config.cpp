#include "stdafx.h"

#include "Config.h"
#include "Core.h"

namespace Config {
  Core *m_pConfig;
  void SetRadix(int r)
  {
    switch(r) {
    case 2:
    case 3:
      m_pConfig = new Config::Ternary::Core();
      break;
    default:
      throw "Unknown Raidx";
    }
  }
  int RadixDigits(int term) {return m_pConfig->RadixDigits(term);}
  int BandSum(int term) {return m_pConfig->BandSum(term);}
}