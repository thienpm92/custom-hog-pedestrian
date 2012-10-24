//
//  main.cpp
//  OpenCVTutorial
//
//  Created by Saburo Okita on 13/8/12.
//  Copyright (c) 2012 Saburo Okita. All rights reserved.
//

/* Standard headers */
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>

/* OpenCV headers */
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/stitching/stitcher.hpp>

/* Boost headers */
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include "ImageProcessor.h"
#include "HOGProcessor.h"
#include "Utils.h"
#include "Stopwatch.h"
#include "CustomHOG.h"


using namespace std;
using namespace boost::filesystem3;
using namespace cv;
using namespace cv::detail;


int main(int argc, const char * argv[]) {
        
    //Mat img = imread( "/Users/sub/Desktop/DSC04075.jpg" );
    // Mat templ = imread( "/Users/sub/Desktop/template.jpg" );
    

    Mat img = imread( "/Users/sub/Desktop/pedestrian wave.JPG" );
    hog::CustomHOG custom_hog;
    
    //custom_hog.extractAndSaveHOGFeatures( "/Users/sub/dataset/pos/", "/Users/sub/dataset/neg/", "/Users/sub/Desktop/", 1000, 1000 );
    
    //Mat training_data = custom_hog.loadFeaturesFromFile( "/Users/sub/Desktop", 1000, 1000 );custom_hog.trainSVM( training_data, 1000, 1000, "/Users/sub/Desktop/svm_file_new.txt" );
    
    CvSVM svm;
    svm.load( "/Users/sub/Desktop/svm_file_new.txt" );
    custom_hog.detect( svm, img );
    
    cout << "Done " << Utils::getCurrentTime() << endl;
    return 0;
}

