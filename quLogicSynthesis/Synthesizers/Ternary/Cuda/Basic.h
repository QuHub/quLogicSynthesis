#pragma once
#include "../Basic.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaSequence.h"
#include "cuda_debug.h"

void SynthesizeKernel(CudaSequence *pcuSeq);

namespace Synthesizer {
  namespace Ternary {
    namespace Cuda {
      class Basic : public Core {
        CudaSequence m_cuSeq;
        CudaSequence* m_pcuPacket;
        int m_nTotalTransferTerms;
        int m_nTotalTransferGates;
        int m_nTerms;

      public:
        Basic(int nBits) : Core(nBits) { }


        void Process()
        {
          Process(1);
          printf("Here we are in the middle of this");
          Process(0);
        }

        void Process(int device)
        {
          // NOTE: This is essential for Parallel Nsight debugging, since GPU1 is used to debug the
          // code, while GPU0 is used for the display.
          cudaSetDevice(device);  
          TransferToDevice();

          SynthesizeKernel(m_pcuPacket);
          //cudaThreadSynchronize();

          // check for error
          cudaError_t error = cudaGetLastError();
          if(error != cudaSuccess)
          {
            printf("My CUDA error: %s\n", cudaGetErrorString(error));
          }

          TransferFromDevice();
        }

        void InitTransferPacket()
        {
          m_nTerms = m_cuSeq.m_nTerms = Helper::BitsToTerms(m_nBits);
          m_cuSeq.m_nBits = m_nBits;
          m_cuSeq.m_nSequences = (int)m_Sequences.size();
          m_cuSeq.m_nMaxGates = MAX_GATES;

          m_nTotalTransferTerms = m_nTerms  * m_cuSeq.m_nSequences;
          m_nTotalTransferGates = MAX_GATES * m_cuSeq.m_nSequences; 
        }

        void TransferToDevice()
        {
          InitTransferPacket();
          AllocateTransferMemory();

          // Copy input and output buffers to device
          CS( cudaMalloc( (void**)&m_pcuPacket, sizeof(CudaSequence)) );
          CS( cudaMemcpy(m_pcuPacket, &m_cuSeq, sizeof(m_cuSeq), cudaMemcpyHostToDevice) );

          for(int i=0; i<m_cuSeq.m_nSequences; i++) {
            CopyMemory(&m_cuSeq.m_pIn[i*m_nTerms],  m_Sequences[i]->InputForRadix(),  by(m_nTerms) );
            CopyMemory(&m_cuSeq.m_pOut[i*m_nTerms], m_Sequences[i]->OutputForRadix(), by(m_nTerms) );
          }

          CS( cudaMemcpy(m_cuSeq.m_cuIn,  m_cuSeq.m_pIn,  by(m_nTotalTransferTerms), cudaMemcpyHostToDevice ));
          CS( cudaMemcpy(m_cuSeq.m_cuOut, m_cuSeq.m_pOut, by(m_nTotalTransferTerms), cudaMemcpyHostToDevice ));
        }

        void TransferFromDevice()
        {
          // Copy input and output buffers to device
          CS( cudaMemcpy(m_cuSeq.m_pnGates,    m_cuSeq.m_cuGates,     by(m_cuSeq.m_nSequences),  cudaMemcpyDeviceToHost) );
          CS( cudaMemcpy(m_cuSeq.m_pControl,   m_cuSeq.m_cuControl,   by(m_nTotalTransferGates), cudaMemcpyDeviceToHost) );
          CS( cudaMemcpy(m_cuSeq.m_pOperation, m_cuSeq.m_cuOperation, m_nTotalTransferGates,     cudaMemcpyDeviceToHost) );
          CS( cudaMemcpy(m_cuSeq.m_pTarget,    m_cuSeq.m_cuTarget,    m_nTotalTransferGates,     cudaMemcpyDeviceToHost) );

          for(int i=0; i<m_cuSeq.m_nSequences; i++) {
            int nGates = m_Sequences[i]->m_nGates = m_cuSeq.m_pnGates[i];
            LPBYTE pDst = m_Sequences[i]->m_pTarget;
            LPBYTE pSrc = &m_cuSeq.m_pTarget   [i*MAX_GATES];
            ZeroMemory(m_Sequences[i]->m_pControl, by(nGates));
            ZeroMemory(m_Sequences[i]->m_pTarget, nGates);
            ZeroMemory(m_Sequences[i]->m_pOperation, nGates);
            CopyMemory(m_Sequences[i]->m_pControl,   &m_cuSeq.m_pControl  [i*MAX_GATES], by(nGates));
            CopyMemory(m_Sequences[i]->m_pTarget,    &m_cuSeq.m_pTarget   [i*MAX_GATES], nGates);
            CopyMemory(m_Sequences[i]->m_pOperation, &m_cuSeq.m_pOperation[i*MAX_GATES], nGates);
          }

          FreeTransferMemory();
        }

        LPINT AllocateMemory(int size)
        {
          LPVOID ptr = VirtualAlloc(NULL,size , MEM_COMMIT, PAGE_READWRITE);
          if (ptr == NULL) {
            DWORD err= GetLastError();
            throw("Error Allocating Memory");
          }

          return (LPINT)ptr;
        }

        void AllocateTransferMemory()
        {
          m_cuSeq.m_pIn         = AllocateMemory(by(m_nTotalTransferTerms));
          m_cuSeq.m_pOut        = AllocateMemory(by(m_nTotalTransferTerms));
          m_cuSeq.m_pnGates     = AllocateMemory(m_cuSeq.m_nTerms * sizeof(int));
          m_cuSeq.m_pControl    = AllocateMemory(by(m_nTotalTransferGates));
          m_cuSeq.m_pOperation  = (LPBYTE)AllocateMemory(m_nTotalTransferGates);
          m_cuSeq.m_pTarget     = (LPBYTE)AllocateMemory(m_nTotalTransferGates);

          CS( cudaMalloc( (void**)&m_cuSeq.m_cuIn, by(m_nTotalTransferTerms)) );
          CS( cudaMalloc( (void**)&m_cuSeq.m_cuOut, by(m_nTotalTransferTerms)) );
          CS( cudaMalloc( (void**)&m_cuSeq.m_cuControl,by(m_nTotalTransferGates)) );
          CS( cudaMalloc( (void**)&m_cuSeq.m_cuTarget, m_nTotalTransferGates) );
          CS( cudaMalloc( (void**)&m_cuSeq.m_cuOperation, m_nTotalTransferGates) );
          CS( cudaMalloc( (void**)&m_cuSeq.m_cuGates, m_cuSeq.m_nSequences * sizeof(int)) );
        }

        void FreeTransferMemory() 
        {
          VirtualFree(m_cuSeq.m_pIn, by(m_nTotalTransferTerms), MEM_RELEASE);
          VirtualFree(m_cuSeq.m_pOut, by(m_nTotalTransferTerms), MEM_RELEASE);
          VirtualFree(m_cuSeq.m_pnGates, m_cuSeq.m_nTerms * sizeof(int), MEM_RELEASE);
          VirtualFree(m_cuSeq.m_pControl, by(m_nTotalTransferGates), MEM_RELEASE);
          VirtualFree(m_cuSeq.m_pOperation, m_nTotalTransferGates, MEM_RELEASE);
          VirtualFree(m_cuSeq.m_pTarget, m_nTotalTransferGates, MEM_RELEASE);

          cudaFree(m_cuSeq.m_cuIn);
          cudaFree(m_cuSeq.m_cuOut);
          cudaFree(m_cuSeq.m_cuTarget);
          cudaFree(m_cuSeq.m_cuOperation);
          cudaFree(m_cuSeq.m_cuControl);
          cudaFree(m_cuSeq.m_cuGates);
          CS( cudaFree(m_pcuPacket) );
        }
      };
    }
  }
}