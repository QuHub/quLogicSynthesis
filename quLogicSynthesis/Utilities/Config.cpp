#include "stdafx.h"

#include "Config.h"
#include "ConfigTernary.h"

namespace Config {
  Core *m_pConfig;
  void SetRadix(int r, int nBits)
  {
    switch(r) {
    case 2:
    case 3:
      m_pConfig = new Config::Ternary::Core(nBits);
      break;
    default:
      throw "Unknown Raidx";
    }
  }
  int RadixDigits(int term) {return m_pConfig->RadixDigits(term);}
  int BandSum(int term) {return m_pConfig->BandSum(term);}
  int Radix() {return m_pConfig->m_Radix;}
  int RadixBits() {return m_pConfig->m_RadixBits;}
  int RadixMask() {return m_pConfig->m_RadixMask;}
  int Bits() {return m_pConfig->m_nBits;}
}