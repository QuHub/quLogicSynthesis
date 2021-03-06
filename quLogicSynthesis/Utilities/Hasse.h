#pragma warning(disable:4996)
#pragma once
#include "stdafx.h"

namespace Utility {
  namespace Ternary {
    class Hasse {
    public:
      int m_nBits;
      int m_nTerms;
      int m_nBands;
      vector<int> *m_pBands;

    public:
      Hasse(int nBits)
      {
        // Allocate space for bands
        m_nBits = nBits;
        m_nTerms = (int)pow((double)Config::Radix(), nBits);
        m_nBands = (Config::Radix() - 1) * nBits + 1; // see comment below about nBands
        m_pBands = new vector<int>[m_nBands];               

        // Insert each number into its band based on the sum of its digits
        for (int i=0; i<m_nTerms; i++) {
          int termInRadixDigits = Config::RadixDigits(i);
          // Notice that the original number (input index) is added to the band, not the termInRadixDigits.
          // This is needed so that we can match each input term to its output term correctly.
          // See Sequence#GenerateOutput().
          m_pBands[Config::BandSum(termInRadixDigits)].push_back(i);   
        }
      }

      // Serialize Hasse sequence and copy it into the buffer p
      // WARNING: returned pointer needs to be freed by caller.
      int* GetSequence()
      {
        for (int i=0; i<m_nBands; i++) {
          random_shuffle(m_pBands[i].begin(), m_pBands[i].end());
        }

        int* p = new int[m_nTerms];       // NOTE: To avoid memory leaks, this needs to be freed by the caller.
        int* s = p;
        for (int i=0; i<m_nBands; i++) {
          for (int j=0; j< (int)m_pBands[i].size(); j++) {
            *p++ = m_pBands[i][j];
          }
        }
        return s;
      }

      ~Hasse()
      {
        delete m_pBands;
      }
    };

  }
}

// **** nBands:
// A Radix (x) Hasse diagram for n variables we will have k Levels (bands), where the
// sum of digits for each band increases by one from band to band, with 
// the maximum sum at the top consists of (Radix - 1) repeated n times.  So:
//   Number of bands = (Radix-1)*n + 1 (the 1 is for 0000...00)
//   Example: For 2 bits:  
//             Binary                                 Ternary
//       Band                Digit Sum           Band                Digit Sum              
//       {000},                 0                {00},                 0     
//       {010, 100, 001},       1                {01, 10},             1     
//       {011, 110, 101},       2                {02, 11, 20},         2     
//       {111},                 3                {12, 21},             3     
//                                               {22}                  4     
//        Total: 1*3 + 1 = 4 bands.              Total: 2*2 + 1 = 5 bands