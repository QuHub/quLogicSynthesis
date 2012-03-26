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

namespace Rand
{
  Rand *m_pRandom;

  Rand::Rand(void)
  {
    m_hMutex = ::CreateMutexA(NULL, false, NULL);
    Fill(100000);
  }

  Rand::~Rand()
  {
    ::ReleaseMutex(m_hMutex);
  }

  DWORD Rand::Run(LPVOID args) 
  { 
    while(true) {
      if( (m_numbers.size() < 50000) || m_forceRefill) 
          Fill();

      Sleep(100);
    }

    return true;
  }

  void Rand::ReFill()
  {
    m_forceRefill = true;
  }

  using namespace System::Runtime::InteropServices;
  void Rand::Fill(int fill)
  {
    Lock();
    WebRequest^ req = WebRequest::Create("http://www.randomserver.dyndns.org/client/random.php?type=LIN&a=0&b=1&file=0&n=" + fill);
    HttpWebResponse^ resp = dynamic_cast<HttpWebResponse^>(req->GetResponse());

    StreamReader^ reader = gcnew StreamReader(resp->GetResponseStream());
    // Skip first two lines
    String^ line = reader->ReadLine();
    line = reader->ReadLine();

    line = reader->ReadLine();
    IntPtr ip = Marshal::StringToBSTR(line);
    char* str = static_cast<char*>(ip.ToPointer());

    while (line)
    {
      Extract(line);
      line = reader->ReadLine();
    }
    Release();
  }

  double Rand::Double()
  { 
    if(m_numbers.size() < 100) 
      throw "We are running out"; 
    
    Lock(); 
    double n = m_numbers.front(); 
    m_numbers.pop(); 
    Release();
    return n;
  }
#pragma managed  // This stupid thing is necessary to be able to use regex matches.
  void Rand::Extract(String^ line)
  {
    MatchCollection^ m = Regex::Matches(line, "(\\d.\\d*)");
    m_numbers.push(Convert::ToDouble(m[0]->Result("$1")));
    m_numbers.push(Convert::ToDouble(m[1]->Result("$1")));
  }

}