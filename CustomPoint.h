//
//  CustomPoint.h
//  OpenCVTutorial
//
//  Created by Saburo Okita on 23/10/12.
//  Copyright (c) 2012 Saburo Okita. All rights reserved.
//

#ifndef OpenCVTutorial_CustomPoint_h
#define OpenCVTutorial_CustomPoint_h

#include <vector>
using namespace std;

class CustomPoint {
    public:
        int x;
        int y;
        float scale;
        float weight;
    
        CustomPoint();
        CustomPoint( int x, int y, float scale, float weight );

        bool equals( CustomPoint &point );
        bool within(int a, int b, double tolerance);
        bool equalsWithTolerance( CustomPoint &point, double tolerance );
        bool existsIn( vector<CustomPoint> &points );
};

#endif
