//
//  Utils.cpp
//  OpenCVTutorial
//
//  Created by Saburo Okita on 10/9/12.
//  Copyright (c) 2012 Saburo Okita. All rights reserved.
//

#include "Utils.h"
#include <string>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <ctime>

string Utils::getCurrentTime() {
    stringstream string_stream;
    time_t t = time( 0 );
    struct tm * now = localtime( &t );
    
    string_stream   << now->tm_mday << "/"
                    << now->tm_mon  << "/"
                    << (now->tm_year + 1900) << " "
                    << now->tm_hour << ":" << now->tm_min << ":" << now->tm_sec ;
    
    return string_stream.str();
}

vector<path> Utils::listFiles( const char * pathname ) {
    path directory_path( pathname );
    if( exists( directory_path ) == false ) {
        string exception_message = "[ERROR] ";
        exception_message.append( directory_path.string() );
        throw exception_message.c_str();
    }
    
    vector<path> files;
    copy( directory_iterator( directory_path ), directory_iterator(), back_inserter( files )  );
    
    return files;
}