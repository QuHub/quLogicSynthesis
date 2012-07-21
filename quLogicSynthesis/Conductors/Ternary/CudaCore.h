#pragma once
#include "ConductorCore.h"
#include "../../Generators/GeneratorCore.h"
#include "../../Synthesizers/SynthesizerCore.h"
#include "../../Sequence.h"
#include "../../Synthesizers/Ternary/Cuda/CudaSequence.h"
#include "../../Utilities/Rand.h"
#include "../../Utilities/Thread.h"
#include "Factory.h"

using namespace System;
using namespace System::IO;

namespace Conductor {
  class CudaCore : public CThread
  {
    DWORD Run(LPVOID args)
    {
      Factory *factory = (Factory *)args;
      DWORD threadId = factory->m_threadId;

      while(true) {
        Console::WriteLine("CudaCore[{0}]:: Waiting for Mutex", threadId);
        WaitForSingleObject(factory->m_hMutex, INFINITE);
        Console::WriteLine("CudaCore[{0}]:: Got Mutex", threadId);
        SetEvent(factory->m_hEvent);
        Console::WriteLine("CudaCore[{0}]:: Releaseing Mutex", threadId);
        ReleaseMutex(factory->m_hMutex);
      }
    }
  };
}