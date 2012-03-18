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
          AllocateTransferMemory(m_cuSeq);

          // Copy input and output buffers to device
          CS( cudaMalloc( (void**)&m_pcuPacket, sizeof(CudaSequence)) );
          CS( cudaMemcpy(m_pcuPacket, &m_cuSeq, sizeof(m_cuSeq), cudaMemcpyHostToDevice) );

          int vectorLength = m_nTerms * sizeof(int);
          for(int i=0; i<m_cuSeq.m_nSequences; i++) {
            CopyMemory(&m_cuSeq.m_pIn[i*m_nTerms], m_Sequences[i]->InputForRadix(), vectorLength );
            CopyMemory(&m_cuSeq.m_pOut[i*m_nTerms], m_Sequences[i]->OutputForRadix(), vectorLength );
          }

          CS( cudaMemcpy(m_cuSeq.m_cuIn, m_cuSeq.m_pIn, by(m_nTotalTransferTerms), cudaMemcpyHostToDevice ));
          CS( cudaMemcpy(m_cuSeq.m_cuOut, m_cuSeq.m_pOut, by(m_nTotalTransferTerms), cudaMemcpyHostToDevice ));
        }

        void Process(int device)
        {
          // NOTE: This is essential for Parallel Nsight debugging, since GPU1 is used to debug the
          // code, while GPU0 is used for the display.
          cudaSetDevice(device);  
          InitTransferPacket();
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

        void TransferFromDevice()
        {
          // Copy input and output buffers to device
          int outputBufferBytes = Sequence::OutputBufferBytes * m_cuSeq.m_nSequences;
          CS( cudaMemcpy(m_cuSeq.m_pnGates, m_cuSeq.m_cuGates, m_cuSeq.m_nSequences * sizeof(int), cudaMemcpyDeviceToHost) );
          CS( cudaMemcpy(m_cuSeq.m_pControl, m_cuSeq.m_cuControl, outputBufferBytes, cudaMemcpyDeviceToHost) );
          CS( cudaMemcpy(m_cuSeq.m_pOperation, m_cuSeq.m_cuOperation, outputBufferBytes, cudaMemcpyDeviceToHost) );
          CS( cudaMemcpy(m_cuSeq.m_pTarget, m_cuSeq.m_cuTarget, outputBufferBytes, cudaMemcpyDeviceToHost) );

          for(int i=0; i<m_cuSeq.m_nSequences; i++) {
            int nGates = m_Sequences[i]->m_nGates = m_cuSeq.m_pnGates[i];
            int nBytes = nGates * sizeof(int);
            CopyMemory(m_Sequences[i]->m_pControl, &m_cuSeq.m_pControl[i*m_cuSeq.m_nMaxGates], nBytes);
            CopyMemory(m_Sequences[i]->m_pTarget, &m_cuSeq.m_pTarget[i*m_cuSeq.m_nMaxGates], nBytes);
            CopyMemory(m_Sequences[i]->m_pOperation, &m_cuSeq.m_pOperation[i*m_cuSeq.m_nMaxGates], nBytes);
          }

          CS( cudaFree(m_pcuPacket) );
          FreeTransferMemory(m_cuSeq);
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

        void AllocateTransferMemory(CudaSequence &seq)
        {
          int inputBufferBytes = seq.m_nTerms * seq.m_nSequences * sizeof(int);
          int outputBufferBytes = Sequence::OutputBufferBytes * seq.m_nSequences;

          seq.m_pIn         = AllocateMemory(inputBufferBytes);
          seq.m_pOut        = AllocateMemory(inputBufferBytes);
          seq.m_pnGates     = AllocateMemory(seq.m_nTerms * sizeof(int));
          seq.m_pControl = AllocateMemory(outputBufferBytes);
          seq.m_pOperation = AllocateMemory(outputBufferBytes);
          seq.m_pTarget = AllocateMemory(outputBufferBytes);

          CS( cudaMalloc( (void**)&seq.m_cuIn, inputBufferBytes) );
          CS( cudaMalloc( (void**)&seq.m_cuOut, inputBufferBytes) );
          CS( cudaMalloc( (void**)&seq.m_cuTarget, outputBufferBytes) );
          CS( cudaMalloc( (void**)&seq.m_cuOperation, outputBufferBytes) );
          CS( cudaMalloc( (void**)&seq.m_cuControl,outputBufferBytes) );
          CS( cudaMalloc( (void**)&seq.m_cuGates, seq.m_nSequences * sizeof(int)) );
        }

        void FreeTransferMemory(CudaSequence &seq) 
        {
          int inputBufferBytes = seq.m_nTerms * seq.m_nSequences * sizeof(int);
          int outputBufferBytes = Sequence::OutputBufferBytes * seq.m_nSequences;

          VirtualFree(seq.m_pIn, inputBufferBytes, MEM_RELEASE);
          VirtualFree(seq.m_pOut, inputBufferBytes, MEM_RELEASE);
          VirtualFree(seq.m_pnGates, seq.m_nTerms * sizeof(int), MEM_RELEASE);
          VirtualFree(seq.m_pControl, outputBufferBytes, MEM_RELEASE);
          VirtualFree(seq.m_pOperation, outputBufferBytes, MEM_RELEASE);
          VirtualFree(seq.m_pTarget, outputBufferBytes, MEM_RELEASE);

          cudaFree(seq.m_cuIn);
          cudaFree(seq.m_cuOut);
          cudaFree(seq.m_cuTarget);
          cudaFree(seq.m_cuOperation);
          cudaFree(seq.m_cuControl);
          cudaFree(seq.m_cuGates);
        }

        //void TransferToCuda(CudaSequence &seq)
        //{
        //  int bufferSizeBytes = seq.m_nTerms * seq.m_nSequences * sizeof(int);
        //  int outputBlockSize = 200*1024*seq.m_nSequences;
        //  CudaSequence *pcuSeq;

        //  // NOTE: This is essential for Parallel Nsight debugging, since GPU1 is used to debug the
        //  // code, while GPU0 is used for the display.
        //  cudaSetDevice(1);  

        //  CS( cudaMalloc( (void**)&pcuSeq, sizeof(CudaSequence)) );
        //  CS( cudaMalloc( (void**)&seq.m_cuIn, bufferSizeBytes) );
        //  CS( cudaMalloc( (void**)&seq.m_cuOut, bufferSizeBytes) );
        //  CS( cudaMalloc( (void**)&seq.m_cuTarget, outputBlockSize) );
        //  CS( cudaMalloc( (void**)&seq.m_cuOperation, outputBlockSize) );
        //  CS( cudaMalloc( (void**)&seq.m_cuControl,outputBlockSize) );
        //  CS( cudaMalloc( (void**)&seq.m_cuGates, seq.m_nSequences * sizeof(int)) );

        //  // Copy memory block to CUDA device
        //  CS( cudaMemcpy(seq.m_cuIn, seq.m_pIn, bufferSizeBytes, cudaMemcpyHostToDevice) );
        //  CS( cudaMemcpy(seq.m_cuOut, seq.m_pOut, bufferSizeBytes, cudaMemcpyHostToDevice) );
        //  CS( cudaMemcpy(pcuSeq, &seq, sizeof(seq), cudaMemcpyHostToDevice) );

        //  SynthesizeKernel(pcuSeq);

        //  //// make the host block until the device is finished with foo
        //  //cudaThreadSynchronize();

        //  // check for error
        //  cudaError_t error = cudaGetLastError();
        //  if(error != cudaSuccess)
        //  {
        //    // print the CUDA error message and exit
        //    printf("My CUDA error: %s\n", cudaGetErrorString(error));
        //  }

        //  cudaMemcpy(seq.m_pTarget, seq.m_cuTarget, outputBlockSize, cudaMemcpyDeviceToHost);
        //  cudaMemcpy(seq.m_pControl, seq.m_cuControl, outputBlockSize, cudaMemcpyDeviceToHost);
        //  cudaMemcpy(seq.m_pOperation, seq.m_cuOperation, outputBlockSize, cudaMemcpyDeviceToHost);
        //  cudaMemcpy(seq.m_pnGates, seq.m_cuGates, seq.m_nSequences * sizeof(int), cudaMemcpyDeviceToHost);

        //  // TODO: free up device memory
        //  cudaFree(pcuSeq);
        //  cudaFree(seq.m_cuIn);
        //  cudaFree(seq.m_cuOut);
        //  cudaFree(seq.m_cuTarget);
        //  cudaFree(seq.m_cuOperation);
        //  cudaFree(seq.m_cuControl);
        //  cudaFree(seq.m_cuGates);
        //}
      };
    }
  }
}