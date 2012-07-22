#pragma once
#include "../Basic.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaSequence.h"
#include "cuda_debug.h"

void SynthesizeKernel(int device, cudaStream_t stream, CudaSequence *pcuSeq, int nSequences);

namespace Synthesizer {
  namespace Ternary {
    namespace Cuda {
      class Device: public Core {
        bool m_initialized;
        CudaSequence *m_pcuSeq;
        CudaSequence* m_pcuPacket;
        int m_nTotalTransferTerms;
        int m_nTotalTransferGates;
        int m_nSequences;
        int m_device;
        int m_nTerms;
        cudaStream_t m_stream;

      public:
        Device(int device, int nBits, int nSequences) {
          m_nBits = nBits;
          m_device = device;
          m_nSequences = nSequences;
          m_initialized = false;
          CS(m_device, cudaStreamCreate(&m_stream));
        }

        ~Device() {
          Console::WriteLine("~Device");
          cudaStreamDestroy(m_stream);
          cudaSetDevice(m_device);  
          FreeTransferMemory();
        }

        void Process()
        {
          // NOTE: This is essential for Parallel Nsight debugging, since GPU1 is used to debug the
          // code, while GPU0 is used for the display.
          cudaSetDevice(m_device);  
          Console::WriteLine(String::Format("{0}: Process Device {1}", Helper::StopTimer.getElapsedTime(), m_device ));
          TransferToDevice();
          P(String::Format("{0}: TransferToDevice", Helper::StopTimer.getElapsedTime()));

          SynthesizeKernel(m_device, m_stream, m_pcuPacket, m_pcuSeq->m_nSequences);
          PP(String::Format("{0}: SynthesizeKernel\n", Helper::StopTimer.getElapsedTime()));
        }

        void PostProcess()
        {
          Console::WriteLine(String::Format("{0}: PostProcess Device: {1}\n", Helper::StopTimer.getElapsedTime(), m_device));
          cudaSetDevice(m_device);  
          cudaError_t error = cudaGetLastError();
          if(error != cudaSuccess)
          {
            printf("My CUDA error: %s\n", cudaGetErrorString(error));
          }
          TransferFromDevice();
          P(String::Format("{0}: PostProcess Device: {1}\n", Helper::StopTimer.getElapsedTime(), m_device));
        }

        void InitTransferPacket()
        {
          if(m_initialized) return;

          m_pcuSeq = (CudaSequence*)AllocateMemory(sizeof(CudaSequence));
          m_nTerms = m_pcuSeq->m_nTerms = Helper::BitsToTerms(m_nBits);
          m_pcuSeq->m_nBits = m_nBits;
          m_pcuSeq->m_nSequences = m_nSequences;
          m_pcuSeq->m_nMaxGates = MAX_GATES;

          m_nTotalTransferTerms = m_nTerms  * m_pcuSeq->m_nSequences;
          m_nTotalTransferGates = MAX_GATES * m_pcuSeq->m_nSequences; 

          AllocateTransferMemory();
          m_initialized = true;
        }

        void TransferToDevice()
        {
          InitTransferPacket();

          // Copy input and output buffers to device
          CS(m_device, cudaMemcpyAsync(m_pcuPacket, m_pcuSeq, sizeof(CudaSequence), cudaMemcpyHostToDevice, m_stream) );

          for(int i=0; i<m_pcuSeq->m_nSequences; i++) {
            CopyMemory(&m_pcuSeq->m_pIn[i*m_nTerms],  m_Sequences[i]->InputForRadix(),  by(m_nTerms) );
            CopyMemory(&m_pcuSeq->m_pOut[i*m_nTerms], m_Sequences[i]->OutputForRadix(), by(m_nTerms) );
          }

          P(String::Format("Before cudaMemcpy: {0}\n", Helper::StopTimer.getElapsedTime()));
          CS(m_device, cudaMemcpyAsync(m_pcuSeq->m_cuIn,  m_pcuSeq->m_pIn,  by(m_nTotalTransferTerms), cudaMemcpyHostToDevice, m_stream ));

          P(String::Format("After cudaMemcpy: {0}\n", Helper::StopTimer.getElapsedTime()));
          CS(m_device, cudaMemcpyAsync(m_pcuSeq->m_cuOut, m_pcuSeq->m_pOut, by(m_nTotalTransferTerms), cudaMemcpyHostToDevice, m_stream ));
        }

        void TransferFromDevice()
        {
          // Copy input and output buffers to device
          CS(m_device, cudaMemcpyAsync(m_pcuSeq->m_pnGates,    m_pcuSeq->m_cuNumGates,     by(m_pcuSeq->m_nSequences),  cudaMemcpyDeviceToHost, m_stream) );

#ifdef _DEBUG
          // Transfers back from Cuda are expensive, and there is really no need to copy the entire set of data back, we 
          // just need the number of quantum gates which currently represents the quantum cost.  In the case we use 
          // a different quantum cost, we should return that.

          CS(m_device, cudaMemcpyAsync(m_pcuSeq->m_pControl,   m_pcuSeq->m_cuControl,   by(m_nTotalTransferGates), cudaMemcpyDeviceToHost, m_stream) );
          CS(m_device, cudaMemcpyAsync(m_pcuSeq->m_pGates, m_pcuSeq->m_cuGates, m_nTotalTransferGates,     cudaMemcpyDeviceToHost, m_stream) );
          CS(m_device, cudaMemcpyAsync(m_pcuSeq->m_pTarget,    m_pcuSeq->m_cuTarget,    m_nTotalTransferGates,     cudaMemcpyDeviceToHost, m_stream) );

#endif         
          for(int i=0; i<m_pcuSeq->m_nSequences; i++) {
            int nGates = m_Sequences[i]->m_nGates = m_pcuSeq->m_pnGates[i];
#ifdef _DEBUG
            LPBYTE pDst = m_Sequences[i]->m_pTarget;
            LPBYTE pSrc = &m_pcuSeq->m_pTarget   [i*MAX_GATES];
            ZeroMemory(m_Sequences[i]->m_pControl, by(nGates));
            ZeroMemory(m_Sequences[i]->m_pTarget, nGates);
            ZeroMemory(m_Sequences[i]->m_pGates, nGates);
            CopyMemory(m_Sequences[i]->m_pControl,   &m_pcuSeq->m_pControl  [i*MAX_GATES], by(nGates));
            CopyMemory(m_Sequences[i]->m_pTarget,    &m_pcuSeq->m_pTarget   [i*MAX_GATES], nGates);
            CopyMemory(m_Sequences[i]->m_pGates, &m_pcuSeq->m_pGates[i*MAX_GATES], nGates);
#endif
          }

        }

        LPINT AllocateMemory(int size)
        {
          float* ptr;
//          LPVOID ptr = VirtualAlloc(NULL,size , MEM_COMMIT, PAGE_READWRITE);
          CS(m_device, cudaMallocHost(&ptr, size);
          //if (ptr == NULL) {
          //  DWORD err= GetLastError();
          //  throw("Error Allocating Memory");
          //}

          return (LPINT)ptr;
        }

        void AllocateTransferMemory()
        {
          m_pcuSeq->m_pnGates     = AllocateMemory(m_pcuSeq->m_nSequences * sizeof(int));

          m_pcuSeq->m_pIn         = AllocateMemory(by(m_nTotalTransferTerms));
          m_pcuSeq->m_pOut        = AllocateMemory(by(m_nTotalTransferTerms));

          m_pcuSeq->m_pControl    = AllocateMemory(by(m_nTotalTransferGates));
          m_pcuSeq->m_pGates  = (LPBYTE)AllocateMemory(m_nTotalTransferGates);
          m_pcuSeq->m_pTarget     = (LPBYTE)AllocateMemory(m_nTotalTransferGates);

          size_t free_mem, total_mem;
          cudaMemGetInfo(&free_mem, &total_mem);

          PP( String::Format("Before Alloc: Avail: {0} : Total: {1}\n", free_mem, total_mem));
          CS(m_device, cudaMalloc( (void**)&m_pcuPacket, sizeof(CudaSequence)) );
          CS(m_device, cudaMalloc( (void**)&m_pcuSeq->m_cuNumGates, m_pcuSeq->m_nSequences * sizeof(int)) );

          CS(m_device, cudaMalloc( (void**)&m_pcuSeq->m_cuIn, by(m_nTotalTransferTerms)) );
          CS(m_device, cudaMalloc( (void**)&m_pcuSeq->m_cuOut, by(m_nTotalTransferTerms)) );

          CS(m_device, cudaMalloc( (void**)&m_pcuSeq->m_cuControl,by(m_nTotalTransferGates)) );
          CS(m_device, cudaMalloc( (void**)&m_pcuSeq->m_cuTarget, m_nTotalTransferGates) );
          CS(m_device, cudaMalloc( (void**)&m_pcuSeq->m_cuGates, m_nTotalTransferGates) );

          cudaMemGetInfo(&free_mem, &total_mem);
          PP( String::Format("After Alloc: Avail: {0} : Total: {1}\n", free_mem, total_mem));
        }

        void FreeTransferMemory() 
        {
          cudaFreeHost(m_pcuSeq->m_pnGates);
          cudaFreeHost(m_pcuSeq->m_pIn);
          cudaFreeHost(m_pcuSeq->m_pOut);
          
          cudaFreeHost(m_pcuSeq->m_pControl);
          cudaFreeHost(m_pcuSeq->m_pGates);
          cudaFreeHost(m_pcuSeq->m_pTarget);

          size_t free_mem, total_mem;
          cudaMemGetInfo(&free_mem, &total_mem);
          P( String::Format("Before Free: Avail: {0} : Total: {1}", free_mem, total_mem));

          CS(m_device, cudaFree(m_pcuSeq->m_cuNumGates) );
          CS(m_device, cudaFree(m_pcuPacket) );

          CS(m_device, cudaFree(m_pcuSeq->m_cuIn) );
          CS(m_device, cudaFree(m_pcuSeq->m_cuOut) );

          CS(m_device, cudaFree(m_pcuSeq->m_cuTarget) );
          CS(m_device, cudaFree(m_pcuSeq->m_cuGates) );
          CS(m_device, cudaFree(m_pcuSeq->m_cuControl) );

          cudaFreeHost(m_pcuSeq);

          cudaMemGetInfo(&free_mem, &total_mem);
          P( String::Format("After Free: Avail: {0} : Total: {1}", free_mem, total_mem));
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

        void Process()
        {
          for(int i=0; i<m_nCores; i++) {
            m_pDev[0]->AddSequence(m_Sequences[i]);
            m_pDev[1]->AddSequence(m_Sequences[i+m_nCores]);
          }

          m_pDev[0]->Process();
          m_pDev[1]->Process();
          m_pDev[0]->PostProcess();
          m_pDev[1]->PostProcess();
        }
    };

    }
  }
}