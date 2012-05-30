#pragma once
#include "stdafx.h"
#include "../SynthesizerCore.h"
namespace Synthesizer {
  namespace Ternary {

    class Basic : public Synthesizer::Core {
    public:
      int * m_pOut;
      Basic() {}
      Basic(int* pOut) {
        m_pOut = pOut;
      }
    };
  }
}