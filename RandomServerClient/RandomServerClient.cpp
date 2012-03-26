// RandomServerClient.cpp : main project file.

#include "stdafx.h"
using namespace System;

using namespace RandomServer;

int main(array<System::String ^> ^args)
{
    Console::WriteLine(L"Hello World");
    RandomServer::TrueRandom *p = new RandomServer::TrueRandom();

    return 0;
}
