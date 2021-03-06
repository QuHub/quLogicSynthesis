#pragma once
#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <msclr\marshal.h>
#include <windows.h>

using namespace System;
using namespace System::Collections::Generic;
using namespace std;
using namespace System::IO;

#define SRC "..\\inputs\\"

namespace Utility {
  public ref class FileSrc {
    UINT m_nCount;
    int m_nFiles, m_nSequence, m_nBits, m_nTerms;
    array<String^, 1>^ m_Files;
    StreamReader ^m_sr;
    int *m_pInput;

  public:
    property String^ Name;
    property String^ SeqName;


    //**************
    FileSrc(ULONG nBits, String^ FilePrefix) {
      SeqName = FilePrefix->Replace('*', ' ')->TrimEnd();
      m_nBits = nBits;
      m_nTerms = Helper::BitsToTerms(nBits);

      if ( !Directory::Exists(SRC) )
        throw gcnew Exception(SRC + " Does not exist");

      array<String^, 1>^ files = Directory::GetFiles(SRC, FilePrefix + ".txt");

      m_sr = gcnew StreamReader(files[0]);
      m_pInput = new int[m_nTerms];
    }

    ~FileSrc()
    {
      delete m_pInput;
    }

    //***********************************
    PINT Next()
    {
      String ^s;
      if(m_sr->Peek() >= 0)
        s = m_sr->ReadLine();
      else
        return NULL;

      array<String^>^ list = s->Split(' ');
      PINT p = m_pInput;
      Name = list[0];
      for (int i=0; i<m_nTerms; i++)
        *p++ = Convert::ToUInt32(list[i+1]);

      return m_pInput;
    }
  };
}
