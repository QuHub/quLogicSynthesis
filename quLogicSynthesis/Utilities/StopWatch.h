#pragma once
#include <windows.h>

namespace Utility
{
  typedef struct {
    LARGE_INTEGER start;
    LARGE_INTEGER stop;
  } stopWatch;

  class CStopWatch {
  private:
    stopWatch timer;
    LARGE_INTEGER frequency;

  public:
    double LIToSecs( LARGE_INTEGER & L) {
      return ((double)L.QuadPart /(double)frequency.QuadPart);
    }

    CStopWatch(){
      timer.start.QuadPart=0;
      timer.stop.QuadPart=0;	
      QueryPerformanceFrequency( &frequency );
    }

    void Start( ) {
      QueryPerformanceCounter(&timer.start);
    }

    void Stop( ) {
      QueryPerformanceCounter(&timer.stop);
    }

    void Sample( ) {
      QueryPerformanceCounter(&timer.stop);
    }

    double getElapsedTime() {
      LARGE_INTEGER time;
      Sample();
      time.QuadPart = timer.stop.QuadPart - timer.start.QuadPart;
      return LIToSecs( time) ;
    }
  };
}