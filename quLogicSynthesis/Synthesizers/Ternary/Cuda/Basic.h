#pragma once
#include "../Basic.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaSequence.h"
#include "cuda_debug.h"

void SynthesizeKernel(int device, float* pcuSeq, int nSequences);

namespace Synthesizer {
  namespace Ternary {
    namespace Cuda {
      class Device: public Core {
        bool m_initialized;
        CudaSequence *m_pcuSeq;
        float* m_pcuPacket;
        int m_nTotalTransferTerms;
        int m_nTotalTransferGates;
        int m_nSequences;
        int m_device;
        int m_nTerms;

      public:
        Device(int device, int nBits, int nSequences) {
          m_nBits = nBits;
          m_device = device;
          m_nSequences = nSequences;
          m_initialized = false;
        }

        ~Device() {
          Console::WriteLine("~Device");
          cudaSetDevice(m_device);  
          FreePinnedMemory();
        }


        void Initialize() 
        { 
          m_Sequences.clear(); 
          cudaSetDevice(m_device);  
          cudaDeviceReset();
          cudaDeviceProp deviceProp;
          cudaGetDeviceProperties(&deviceProp, m_device);
          cudaSetDeviceFlags(cudaDeviceMapHost);
          AllocatePinnedMemory();
        }

        void AllocatePinnedMemory()
        {
          CS(m_device, cudaHostAlloc(&m_pcuSeq, sizeof(CudaSequence), cudaHostAllocMapped));
          InitTransferPacket();

          CS(m_device, cudaHostAlloc(&m_pcuSeq->m_pnGates, bytes(m_pcuSeq->m_nSequences), cudaHostAllocMapped));
          CS(m_device, cudaHostAlloc(&m_pcuSeq->m_pIn,     bytes(m_nTotalTransferTerms), cudaHostAllocMapped));
          CS(m_device, cudaHostAlloc(&m_pcuSeq->m_pOut,    bytes(m_nTotalTransferTerms), cudaHostAllocMapped));

          // Map CPU addresses to device addresses
          CS(m_device, cudaHostGetDevicePointer((void **)&m_pcuPacket, (void *)m_pcuSeq, 0));
          CS(m_device, cudaHostGetDevicePointer((void **)&m_pcuSeq->m_cuNumGates, (void *)m_pcuSeq->m_pnGates, 0));
          CS(m_device, cudaHostGetDevicePointer((void **)&m_pcuSeq->m_cuIn, (void *)m_pcuSeq->m_pIn, 0));
          CS(m_device, cudaHostGetDevicePointer((void **)&m_pcuSeq->m_cuOut, (void *)m_pcuSeq->m_pOut, 0));
        }

        void InitTransferPacket()
        {
          m_nTerms = m_pcuSeq->m_nTerms = Helper::BitsToTerms(m_nBits);
          m_pcuSeq->m_nBits = m_nBits;
          m_pcuSeq->m_nSequences = m_nSequences;
          m_pcuSeq->m_nMaxGates = MAX_GATES;

          m_nTotalTransferTerms = m_nTerms  * m_pcuSeq->m_nSequences;
          m_nTotalTransferGates = MAX_GATES * m_pcuSeq->m_nSequences; 
        }

        void FreePinnedMemory()
        {
          CS(m_device, cudaFreeHost(m_pcuSeq->m_pnGates));
          CS(m_device, cudaFreeHost(m_pcuSeq->m_pIn));
          CS(m_device, cudaFreeHost(m_pcuSeq->m_pOut));
          CS(m_device, cudaFreeHost(m_pcuSeq));
        }

        void Process()
        {
          // NOTE: This is essential for Parallel Nsight debugging, since GPU1 is used to debug the
          // code, while GPU0 is used for the display.
          cudaSetDevice(m_device);  
          P(String::Format("{0}: Process Device {1}", Helper::StopTimer.getElapsedTime(), m_device ));

          for(int i=0; i<m_pcuSeq->m_nSequences; i++) {
            CopyMemory(&m_pcuSeq->m_pIn[i*m_nTerms],  m_Sequences[i]->InputForRadix(),  bytes(m_nTerms) );
            CopyMemory(&m_pcuSeq->m_pOut[i*m_nTerms], m_Sequences[i]->OutputForRadix(), bytes(m_nTerms) );
          }

          SynthesizeKernel(m_device, m_pcuPacket, m_pcuSeq->m_nSequences);
          P(String::Format("{0}: SynthesizeKernel\n", Helper::StopTimer.getElapsedTime()));
        }

        void PostProcess()
        {
          cudaSetDevice(m_device);  
          cudaDeviceSynchronize();
          P(String::Format("{0}: PostProcess Device: {1}\n", Helper::StopTimer.getElapsedTime(), m_device));
          for(int i=0; i<m_pcuSeq->m_nSequences; i++) 
            m_Sequences[i]->m_nGates = m_pcuSeq->m_pnGates[i];
        }

      };

      class Basic : public Core {
      public:
        Device *m_pDev[2];
        int m_nCores;
        Basic(int nBits) : Core(nBits) {
          m_nCores = 512;
          m_pDev[0] = new Device(0, m_nBits, m_nCores);
          m_pDev[1] = new Device(1, m_nBits, m_nCores);
        }

        ~Basic()
        {
          Console::WriteLine("~Basic");
          delete m_pDev[0];
          delete m_pDev[1];
        }

        void Initialize()
        {
          m_pDev[0]->Initialize();
          m_pDev[1]->Initialize();
        }

        void Process()
        {
          m_pDev[0]->m_Sequences.clear();
          m_pDev[1]->m_Sequences.clear();

          for(int i=0; i<m_nCores; i++) {
            m_pDev[0]->AddSequence(m_Sequences[i]);
            m_pDev[1]->AddSequence(m_Sequences[i+m_nCores]);
          }

          m_pDev[0]->Process();
          m_pDev[1]->Process();
        }

        void PostProcess()
        {
          m_pDev[0]->PostProcess();
          m_pDev[1]->PostProcess();
        }
    };

    }
  }
}