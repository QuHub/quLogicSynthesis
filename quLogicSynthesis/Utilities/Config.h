#pragma once

namespace Config {
  class Core {
  public:
    int m_RadixBits;
    int m_Radix;
    int m_RadixMask;
    int m_nBits;

    Core(int nBits) {
      m_nBits = nBits;
    }

    int RadixDigits(int term)
    {
      int t=0;

      for (int i=0; i<m_nBits; i++) {
        t += (term % m_Radix) << (m_RadixBits*i);
        term /= m_Radix;
      }

      return t;
    }  

    int BandSum(int term)
    {
      int nCount=0;
      // 8: 8 bits per byte...
      for (int i=0; i< (int)(8*sizeof(int))/m_RadixBits; i++) {
        nCount += term & m_RadixMask;         
        term >>= m_RadixBits;
      }
      return nCount;
    }
  };

  void SetRadix(int r, int nBits);
  int RadixDigits(int term);
  int BandSum(int term);
  int Radix();
  int RadixBits();
  int Bits();
  int RadixMask();
  static int Date= 23;
}
