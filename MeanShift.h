//
//  MeanShift.h
//  OpenCVTutorial
//
//  Created by Saburo Okita on 23/10/12.
//  Copyright (c) 2012 Saburo Okita. All rights reserved.
//

#ifndef OpenCVTutorial_MeanShift_h
#define OpenCVTutorial_MeanShift_h

#include <vector>
#include "CustomPoint.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class MeanShift {
    private:
        vector<CustomPoint> points;
        vector<CustomPoint> modes;
        double sigmaX;
        double sigmaY;
        double sigmaS;
        int pointIndex;
        int maxIteration;
        
        CustomPoint computeMode( CustomPoint y );
        Mat computeMatrixHH( CustomPoint y );
        Mat computeMatrixH( double scale );
        double computeWeight( int i, CustomPoint point);
        double computeHNorm( Mat& H );
        double computeMahalanaboisDistance( CustomPoint y, CustomPoint yi, Mat &Hi );
        double contributePoint( CustomPoint y, CustomPoint yi, Mat &Hi );
        double weightTransformation( double weight, int c );
        
    public:
        MeanShift( vector<CustomPoint> &points, double sigma_x, double sigma_y, double sigma_s, int point_index, int max_iteration );
        void calc( vector<CustomPoint> &modes );
};

#endif
