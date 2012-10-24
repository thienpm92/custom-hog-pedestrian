//
//  CustomPoint.cpp
//  OpenCVTutorial
//
//  Created by Saburo Okita on 23/10/12.
//  Copyright (c) 2012 Saburo Okita. All rights reserved.
//

#include "CustomPoint.h"


CustomPoint::CustomPoint() {
}


CustomPoint::CustomPoint( int x, int y, float scale, float weight ) {
    this->x = x;
    this->y = y;
    this->scale = scale;
    this->weight = weight;
}


bool CustomPoint::equals( CustomPoint &point ){
    return ( x == point.x && y == point.y && scale == point.scale );
}

bool CustomPoint::within(int a, int b, double tolerance) {
    return (abs( a - b ) / (double) a) <= tolerance;
}

bool CustomPoint::equalsWithTolerance( CustomPoint &point, double tolerance ){
    return ( within( x, point.x, tolerance ) &&
            within( y, point.y, tolerance ) &&
            within( scale, point.scale, tolerance ) );
}


bool CustomPoint::existsIn( vector<CustomPoint> &points ) {
    for( int i = 0; i < points.size(); i++ ) {
        if( equalsWithTolerance( points[i], 0.01 ) )
            return true;
    }
    return false;
}