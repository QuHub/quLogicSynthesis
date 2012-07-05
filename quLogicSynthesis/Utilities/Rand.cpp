#include "stdafx.h"
#pragma once
#include <vcclr.h>
using namespace System::Reflection;
namespace Rand
{
  gcroot<Random^> m_rnd = gcnew Random();
  double Double() {return m_rnd->NextDouble();}
  int Integer(int n) {return m_rnd->Next(n);}
};