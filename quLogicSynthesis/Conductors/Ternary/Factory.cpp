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
      // We want to own the Mutex before we let the main thread kick off the CUDA cores
      WaitForSingleObject(m_hMutex, INFINITE);
      SetEvent(m_hEvent);  

      m_threadId = GetCurrentThreadId();
      Console::WriteLine("Factory[{0}]:: Got Mutex", m_threadId);

      int j=0;
      try
      {
          while(j++<100) {
           Sequence* sequences[NUM_CUDA_BLOCKS];
           for(int i=0; i<100; i++)  {
             sequences[i] = pAlgo->m_pGenerator->GetSequence();
           }

           pAlgo->m_pGenerator->ReleaseSequences();
          }
      }
      catch (SEHException^ e)
      {
        Console::WriteLine("Memory exception");
      	
      }
      ReleaseMutex(m_hMutex);
      Console::WriteLine("Factory[{0}]:: Releasing Mutex", m_threadId);
      return 0;
    }
}