#pragma once

namespace Config {
  class Core {
  public:
    int m_RadixBits;
    int m_Radix;
    int m_RadixMask;

    int RadixDigits(int term)
    {
      int t=0;

      for (int i=0; i<m_RadixBits; i++) {
        t += (term % m_Radix) << (m_RadixBits*i);
        term /= m_RadixMask;
      }

      return t;
    }  

    int BandSum(int term)
    {
      int nCount=0;
      for (int i=0; i< (int)(8*sizeof(int))/m_RadixBits; i++) {
        nCount += term & m_RadixMask;         
        term >>= m_RadixBits;
      }
      return nCount;
    }
  };

  void SetRadix(int r);
  int RadixDigits(int term);
  int BandSum(int term);
  int Radix();
  static int Date= 23;
}
