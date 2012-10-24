//
//  HOGProcessor.h
//  OpenCVTutorial
//
//  Created by Saburo Okita on 28/8/12.
//  Copyright (c) 2012 Saburo Okita. All rights reserved.
//

#ifndef OpenCVTutorial_HOGProcessor_h
#define OpenCVTutorial_HOGProcessor_h

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class HOGProcessor {
    private:
        Size cellSize;
        Size blockSize;
        float stepOverlap;
    
    public:
        HOGProcessor();
        ~HOGProcessor();
    
        void setParams( Size cell_size, Size block_size, float step_overlap );
        Rect mergeRects( vector<Rect> rectangles );
        void calculateHOGBlock( Rect block, Mat& hog_block, vector<Mat>& integrals, Size cell_size, int normalization );
        void calculateHOGRect( Rect cell, Mat& hog_cell, vector<Mat>& integrals, int normalization );
        vector<Mat> calculateIntegralHOG( Mat& image );
        Mat calculateHOGWindow( vector<Mat> integrals, Rect window, int normalization );
        Mat calculateHOGWindow( Mat& image, vector<Mat> integrals, Rect window, int normalization );
    
        Mat train64x128( const char * filepath, Size window, int no_of_samples, const char * save_filename, int normalization );
};

#endif
