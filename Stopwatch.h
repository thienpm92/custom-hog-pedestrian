//
//  Stopwatch.h
//  SVMTrain
//
//  Created by Saburo Okita on 30/7/12.
//  Copyright (c) 2012 Saburo Okita. All rights reserved.
//


#ifndef OpenCVTutorial_Stopwatch_h
#define OpenCVTutorial_Stopwatch_h

#include <iostream>
#include <time.h>

using namespace std;

class Stopwatch {
    private:
        clock_t begin;
        clock_t end;
        double clockDiff( clock_t a, clock_t b );
        
    public:
        Stopwatch();
        ~Stopwatch();
        void start();
        void stop();
        void printElapsedTime( string message = "" );
};

#endif
