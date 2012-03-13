#pragma once
#include "stdafx.h"
#include "Sequence.h"

namespace Generator {
  class Core {
  public:
    virtual Sequence* GetSequence(){return NULL;};
  };
}
