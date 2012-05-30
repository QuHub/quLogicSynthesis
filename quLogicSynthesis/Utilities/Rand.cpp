#include "StdAfx.h"
#include <windows.h>
#include "Rand.h"


using namespace std;
using namespace System;
using namespace System::Collections::Generic;
using namespace System::Text::RegularExpressions;
using namespace System::Text;
using namespace System::Net;
using namespace System::IO;
#using <mscorlib.dll>

using namespace System;
using namespace System::Collections;

#define BufferSize  (sizeof(m_numbers) / sizeof(m_numbers[0]))

namespace Rand
{
  Rand *m_pRandom;

  Rand::Rand(void)
  {
    Fill();
    m_reader = 0;
  }

  DWORD Rand::Run(LPVOID args) 
  { 
    while(true) {
      Fill();
      Sleep(10);
    }

    Console::WriteLine("Why am I exiting");
    return true;
  }

  using namespace System::Runtime::InteropServices;
  void Rand::Fill()
  {
    Console::WriteLine("Fetching True Random Numbers from RandomServer \n");
    WebRequest^ req = WebRequest::Create("http://www.randomserver.dyndns.org/client/random.php?type=LIN&a=0&b=1&file=0&n=" + BufferSize);
    HttpWebResponse^ resp = dynamic_cast<HttpWebResponse^>(req->GetResponse());

    StreamReader^ reader = gcnew StreamReader(resp->GetResponseStream());
    // Skip first two lines
    String^ line = reader->ReadLine();
    line = reader->ReadLine();

    line = reader->ReadLine();

    for(int i=0; i<BufferSize/2; i++) {
      Extract(line, 2*i);
      line = reader->ReadLine();
    }
  }

  double Rand::Double()
  { 
    m_reader = ++m_reader % BufferSize; // circular buffer
    return m_numbers[m_reader];
  }
#pragma managed  // This stupid thing is necessary to be able to use regex matches.
  void Rand::Extract(String^ line, int index)
  {
    MatchCollection^ m = Regex::Matches(line, "(\\d.\\d*)");
    m_numbers[index++] = Convert::ToDouble(m[0]->Result("$1"));
    m_numbers[index++] = Convert::ToDouble(m[1]->Result("$1"));
  }

}