//
//  CustomHOG.h
//  OpenCVTutorial
//
//  Created by Saburo Okita on 12/10/12.
//  Copyright (c) 2012 Saburo Okita. All rights reserved.
//

#ifndef OpenCVTutorial_CustomHog_h
#define OpenCVTutorial_CustomHog_h

/* OpenCV headers */
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

namespace hog {
    class CustomHOG {
        private:
            string inputDirectory;
            string outputDirectory;
        
            Size windowSize;
            Size cellSize;
            Size blockSize;
        
            int noOfBins;
            int featureBlockLen;
            int widthOverlaps;
            int heightOverlaps;
            int totalBlocks;
            
            float stepOverlap;
            /*
            float filterKernel[3][3] = {    {-1,  0,  1},
                                            {-2,  0,  2},
                                            {-1,  0,  1}};
            */
            float filterKernel[1][3] = { {-1, 0, 1} };
            int normalization;
            Mat dxKernel;
            Mat dyKernel;
        
        public:
            CustomHOG();
            ~CustomHOG();
        
            Mat compute( Mat& img );
            Mat compute( Mat& img, Rect window_size );
            Mat convertToTrainingDataMatrix( int size, vector< vector<float> > pos_features, vector< vector<float> > neg_features  );
            Mat loadFeaturesFromFile( string input_path, int no_of_positive, int no_of_negative );
            void extractAndSaveIndividualHOGFeatures( string output_file_path, vector<string> & file_list, int max_samples );
            void extractAndSaveHOGFeatures( string positive_files_path, string negative_files_path, string output_dir, int positive_samples, int negative_samples );
        
            CvSVM trainSVM( Mat & training_data, int no_of_positives, int no_of_negatives, string output_file );
        
            void detect( CvSVM &svm, Mat& img );
            Rect mergeRects( vector<Rect> rects );
        
            Mat preprocessImage( Mat& img );
            vector<Mat> applyEdgeFilter( Mat& img );
        
            Mat calculateMagnitude( Mat &dx, Mat &dy );
            Mat calculateAngle( Mat &dx, Mat &dy );
            vector<Mat> createIntegralBins( Mat &magnitude, Mat &angle );
        
            Mat calculateHOGWindow( Rect window, vector<Mat> integrals );
            void calculateHOGRect( Rect cell, Mat &hog_cell, vector<Mat> &integrals );
            void calculateHOBlock( Rect block, Mat &hog_block, vector<Mat> &integrals );
            Mat createLabelsMatrix( int no_of_positive, int no_of_negative );
    };
    
}

#endif
