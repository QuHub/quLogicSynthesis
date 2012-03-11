#include "stdafx.h"

#include "Config.h"
#include "Core.h"

  static void Config::SetRadix(int r)
  {
    switch(r) {
    case 2:
    case 3:
      m_pConfig = new Config::Ternary::Core();
      break;
    default:
      throw "Unknown Raidx";
    }
  }