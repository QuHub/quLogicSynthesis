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
      public:
        Basic(int nBits) : Core(nBits) { }

        void Process()
        {
          CudaSequence seq;
          seq.m_nSequences = (int)m_Sequences.size();

          // Allocate contiguous buffers.
          int nTerms = seq.m_nTerms = Helper::BitsToTerms(m_nBits);
          int nTotalTerms = nTerms * (int)m_Sequences.size();
          int outputBlockSize = Sequence::OutputBufferSize * seq.m_nSequences;

          // Allocate memory for input and output for all m_sequences, each with m_nTerms

          seq.m_nBits = m_nBits;
          seq.m_pIn         = new int[nTotalTerms];
          seq.m_pOut        = new int[nTotalTerms];
          seq.m_pTarget     = new int[outputBlockSize];
          seq.m_pControl    = new int[outputBlockSize];
          seq.m_pOperation  = new int[outputBlockSize];
          seq.m_pnGates     = new int[nTerms];

          // Copy input and output buffers to device
          for(int i=0; i<seq.m_nSequences; i++) {
            CopyMemory(&seq.m_pIn[i*nTerms],  m_Sequences[i]->InputForRadix(),  nTerms * sizeof(int));
            CopyMemory(&seq.m_pOut[i*nTerms], m_Sequences[i]->OutputForRadix(), nTerms * sizeof(int));
          }

          TransferToCuda(seq);
          // Copy input and output buffers to device
          for(int i=0; i<seq.m_nSequences; i++) {
            int nGates = m_Sequences[i]->m_nGates = seq.m_pnGates[i];
            CopyMemory(m_Sequences[i]->m_pControl, seq.m_pControl,  nGates * sizeof(int));
            CopyMemory(m_Sequences[i]->m_pOperation, seq.m_pOperation,  nGates * sizeof(int));
            CopyMemory(m_Sequences[i]->m_pTarget, seq.m_pTarget,  nGates * sizeof(int));
          }
        }

        void TransferToCuda(CudaSequence &seq)
        {
          int bufferSizeBytes = seq.m_nTerms * seq.m_nSequences * sizeof(int);
          int outputBlockSize = 200*1024*seq.m_nSequences;
          CudaSequence *pcuSeq;

          // NOTE: This is essential for Parallel Nsight debugging, since GPU1 is used to debug the
          // code, while GPU0 is used for the display.
          cudaSetDevice(1);  

          CS( cudaMalloc( (void**)&pcuSeq, sizeof(CudaSequence)) );
          CS( cudaMalloc( (void**)&seq.m_cuIn, bufferSizeBytes) );
          CS( cudaMalloc( (void**)&seq.m_cuOut, bufferSizeBytes) );
          CS( cudaMalloc( (void**)&seq.m_cuTarget, outputBlockSize) );
          CS( cudaMalloc( (void**)&seq.m_cuOperation, outputBlockSize) );
          CS( cudaMalloc( (void**)&seq.m_cuControl,outputBlockSize) );
          CS( cudaMalloc( (void**)&seq.m_cuGates, seq.m_nSequences * sizeof(int)) );

          // Copy memory block to CUDA device
          CS( cudaMemcpy(seq.m_cuIn, seq.m_pIn, bufferSizeBytes, cudaMemcpyHostToDevice) );
          CS( cudaMemcpy(seq.m_cuOut, seq.m_pOut, bufferSizeBytes, cudaMemcpyHostToDevice) );
          CS( cudaMemcpy(pcuSeq, &seq, sizeof(seq), cudaMemcpyHostToDevice) );

          SynthesizeKernel(pcuSeq);

          //// make the host block until the device is finished with foo
          //cudaThreadSynchronize();

          // check for error
          cudaError_t error = cudaGetLastError();
          if(error != cudaSuccess)
          {
            // print the CUDA error message and exit
            printf("My CUDA error: %s\n", cudaGetErrorString(error));
          }

          cudaMemcpy(seq.m_pTarget, seq.m_cuTarget, outputBlockSize, cudaMemcpyDeviceToHost);
          cudaMemcpy(seq.m_pControl, seq.m_cuControl, outputBlockSize, cudaMemcpyDeviceToHost);
          cudaMemcpy(seq.m_pOperation, seq.m_cuOperation, outputBlockSize, cudaMemcpyDeviceToHost);
          cudaMemcpy(seq.m_pnGates, seq.m_cuGates, seq.m_nSequences * sizeof(int), cudaMemcpyDeviceToHost);

          // TODO: free up device memory
          cudaFree(pcuSeq);
          cudaFree(seq.m_cuIn);
          cudaFree(seq.m_cuOut);
          cudaFree(seq.m_cuTarget);
          cudaFree(seq.m_cuOperation);
          cudaFree(seq.m_cuControl);
          cudaFree(seq.m_cuGates);
        }
      };
    }
  }
}