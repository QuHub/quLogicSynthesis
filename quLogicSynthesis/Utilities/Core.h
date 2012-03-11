#pragma once
#include "Config.h"

namespace Config {
  namespace Ternary {
    const int Radix = 3;
    const int RadixBits = 2;
    const int RadixMask = 3;
    // <summary>
    //  Converts decimal value into digits in the Radix specified by Config::RadixBits.
    //  Example:
    //    2 => 02 (00 10)
    //    3 => 10 (01 00)
    //    4 => 11 (01 01)
    //    5 => 12 (01 10)
    // <inputs>
    //   term: in decimal format
    // <outputs>
    //   term in Radix format (binary representation).
    //

    class Core : public Config::Core {
    public:
      Core() : Config::Core() {
        m_RadixBits = RadixBits;
        m_RadixMask = RadixMask;
        m_Radix = Radix;
      }
    };
  }
}