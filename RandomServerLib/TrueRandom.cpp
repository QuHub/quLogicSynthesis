#include "StdAfx.h"
#include <windows.h>
#include "TrueRandom.h"


using namespace std;
using namespace System;
using namespace System::Collections::Generic;
using namespace System::Text::RegularExpressions;
using namespace System::Text;
using namespace System::Net;
using namespace System::IO;
namespace RandomServer
{
TrueRandom::TrueRandom(void)
{
}

DWORD TrueRandom::Run(LPVOID args) 
{ 
  while(true) {
    Fill();
    Sleep(1000);
    Console::Beep();
  }

  return true;
}

void TrueRandom::Fill()
{
  WebRequest^ req = WebRequest::Create("http://www.randomserver.dyndns.org/client/random.php?type=LIN&a=0&b=1&n=10000&file=0");
  HttpWebResponse^ resp = dynamic_cast<HttpWebResponse^>(req->GetResponse());

  StreamReader^ reader = gcnew StreamReader(resp->GetResponseStream());

  String^ line = reader->ReadLine();
  while (line)
  {
    Match^ m = Regex::Match(line,"(\d+)");

    line = reader->ReadLine();
  }
}
}