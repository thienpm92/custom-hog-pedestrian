//
//  Utils.h
//  OpenCVTutorial
//
//  Created by Saburo Okita on 10/9/12.
//  Copyright (c) 2012 Saburo Okita. All rights reserved.
//

#ifndef OpenCVTutorial_Utils_h
#define OpenCVTutorial_Utils_h

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem3;

class Utils {
    public:
        static string getCurrentTime();
        static vector<path> listFiles( const char * pathname );
};

#endif
