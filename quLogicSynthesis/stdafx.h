// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
using namespace std;

#include "constants.h"
#include "Utilities/Helpers.h"
#include "Utilities/StopWatch.h"
#include "Utilities/ConfigTernary.h"
#include "Utilities/Hasse.h"
#include "Utilities/Results.h"
#include "Utilities/FileSrc.h"
// TODO: reference additional headers your program requires here

#ifdef _DEBUG
  #define P(x) Helper::Print(x)
#else
  #define P(x)
#endif