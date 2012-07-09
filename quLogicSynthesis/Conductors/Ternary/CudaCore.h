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

      while(true) {
        Console::WriteLine("CudaCore:: Waiting for Mutex");
        WaitForSingleObject(factory->m_hMutex, INFINITE);
        Console::WriteLine("CudaCore:: Setting Trigger");
        SetEvent(factory->m_hEvent);
        Console::WriteLine("CudaCore:: Releasing Mutex");
        ReleaseMutex(factory->m_hMutex);
      }
    }
  };
}