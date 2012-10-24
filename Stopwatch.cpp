//
//  Stopwatch.cpp
//  SVMTrain
//
//  Created by Saburo Okita on 30/7/12.
//  Copyright (c) 2012 Saburo Okita. All rights reserved.
//

#include "Stopwatch.h"
#include <iostream>


Stopwatch::Stopwatch() {
    
}

Stopwatch::~Stopwatch() {
    
}

double Stopwatch::clockDiff( clock_t begin, clock_t end ) {
    return double( end - begin ) / CLOCKS_PER_SEC;
}

void Stopwatch::start() {
    begin = clock();
}

void Stopwatch::stop() {
    end = clock();
}

void Stopwatch::printElapsedTime( string message ) {
    if( message.size() == 0 )
        std::cout << "Time elapsed: " << double(clockDiff( begin, end )) << " s" << std::endl;
    else
        std::cout << "Time elapsed for " << message << ": " << double(clockDiff( begin, end )) << " s" << std::endl;
}