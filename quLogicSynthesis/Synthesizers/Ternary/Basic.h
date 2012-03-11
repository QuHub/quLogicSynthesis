#pragma once
#include "../SynthesizerCore.h"
namespace Synthesizer {
  namespace Ternary {

    class Basic : public Core {
    public:
      int * m_pOut;
      Basic() {}
      Basic(int* pOut) {
        m_pOut = pOut;
      }

    };
  }
}