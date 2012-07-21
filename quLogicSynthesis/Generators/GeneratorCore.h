#pragma once
#include "../../Sequence.h"

namespace Generator {
  class Core {
  public:
    virtual Sequence* GetSequence(){return NULL;};
    virtual Sequence* SinglePointCrossOver(Sequence *p1, Sequence *p2, double prob){return NULL;};
    virtual Sequence* TwoPointCrossOver(Sequence *p1, Sequence *p2, double prob){return NULL;};
    virtual void Mutate(Sequence *p1, double prob){};
    virtual void ReleaseSequences(){};
  };
}
