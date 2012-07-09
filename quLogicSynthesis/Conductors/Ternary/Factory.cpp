#pragma once
#include "stdafx.h"
#include "../../Sequence.h"
#include "../../Synthesizers/Ternary/Cuda/CudaSequence.h"
#include "../../Utilities/Rand.h"
#include "../../Utilities/Thread.h"
#include "ShuffleAlgorithm.h"
#include "Factory.h"

namespace Conductor {
    Factory::Factory()
    {
      m_hMutex = CreateMutex(NULL, false, NULL);
      m_hEvent = CreateEvent(NULL, true, false, NULL );
    }

    DWORD Factory::Run(LPVOID arg)
    {
      Shuffle * pAlgo = (Shuffle *)arg;

      WaitForSingleObject(m_hMutex, INFINITE);

      while(true) {
        Sequence* sequences[NUM_CUDA_BLOCKS];
        for(int i=0; i<NUM_CUDA_BLOCKS; i++)  
          sequences[i] = pAlgo->m_pGenerator->GetSequence();

        Console::WriteLine("Factory:: Generated Sequences");
        ReleaseMutex(m_hMutex);
        Console::WriteLine("Factory:: Waiting for Trigger");
        WaitForSingleObject(m_hEvent, INFINITE);
        Console::WriteLine("Factory:: Waiting for Mutex");
        WaitForSingleObject(m_hMutex, INFINITE);
        Console::WriteLine("Factory:: Releaseing Sequences");
        pAlgo->m_pGenerator->ReleaseSequences();
      }
    }
}