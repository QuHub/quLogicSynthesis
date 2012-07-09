#pragma once
#include "ConductorCore.h"

using namespace System;
using namespace System::IO;

namespace Conductor {
  class Factory : public CThread
  {
  public:
    HANDLE m_hMutex;
    HANDLE m_hEvent;
    DWORD Run(LPVOID arg);
    Factory();
  };
}