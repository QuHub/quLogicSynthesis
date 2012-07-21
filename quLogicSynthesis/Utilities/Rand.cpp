#include "stdafx.h"
#pragma once
#include <vcclr.h>
using namespace System::Reflection;
using namespace MathNet::Numerics::Random;
namespace Rand
{
  gcroot<SystemCryptoRandomNumberGenerator^> m_rnd = gcnew SystemCryptoRandomNumberGenerator();
  //gcroot<Xorshift^> m_rnd = gcnew Xorshift();
  void Initialize() {}
  double Double() {return m_rnd->NextDouble();}
  int Integer(int n) {return m_rnd->Next(n);}
};