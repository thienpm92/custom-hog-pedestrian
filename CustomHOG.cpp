//
//  CustomHog.cpp
//  OpenCVTutorial
//
//  Created by Saburo Okita on 12/10/12.
//  Copyright (c) 2012 Saburo Okita. All rights reserved.
//

#include "CustomHOG.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>
#include "CustomPoint.h"
#include "MeanShift.h"

using namespace boost::filesystem3;

namespace hog {
    
    
    CustomHOG::CustomHOG() {
        inputDirectory  = "/Users/sub/dataset/";
        outputDirectory = "/Users/sub/Desktop/";
        
        windowSize  = Size( 64, 128 );
        cellSize    = Size( 8, 8 );
        blockSize   = Size( 2, 2 );
        stepOverlap = 0.5f;
        noOfBins    = 9;
        
        /* each feature block = 36 */
        featureBlockLen = noOfBins * blockSize.height * blockSize.width;
        
        widthOverlaps   = ( ( windowSize.width - cellSize.width * blockSize.width )
                        / ( cellSize.width * stepOverlap ) + 1 );
        
        heightOverlaps  = ( ( windowSize.height - cellSize.height * blockSize.height )
                        / ( cellSize.height * stepOverlap ) + 1 );
        
        totalBlocks     = widthOverlaps * heightOverlaps;
        
        normalization = NORM_L2;
        
        //dxKernel = Mat( 3, 3, CV_32FC1, filterKernel );
        dxKernel = Mat( 1, 3, CV_32FC1, filterKernel );
        dyKernel = dxKernel.t();
    }
    
    CustomHOG::~CustomHOG() {}
    
    
    Mat CustomHOG::convertToTrainingDataMatrix(int size, vector< vector<float> > pos_features, vector< vector<float> > neg_features  ) {
        
        Mat training_data( (int) (pos_features.size() + neg_features.size()), size, CV_32FC1 );
        size_t step = training_data.step1() << 2;
        
        float * ptr;
        int index = 0;
        for( int y = 0; y < pos_features.size(); y++ ) {
            ptr = (float*) (training_data.data + index++ * step );
            for( int x = 0; x < size; x++ )
                ptr[x] = pos_features[y][x];
        }
        
        
        for( int y = 0; y < neg_features.size(); y++ ) {
            ptr = (float*) (training_data.data + index++ * step );
            for( int x = 0; x < size; x++ )
                ptr[x] = neg_features[y][x];
        }
        
        return training_data;
    }
    
    Mat CustomHOG::createLabelsMatrix( int no_of_positive, int no_of_negative ) {
        Mat labels( no_of_positive + no_of_negative, 1, CV_32FC1 );
        Mat roi = labels.rowRange( 0, no_of_positive );
        roi = Scalar(1.0);
        
        roi = labels.rowRange( no_of_positive, no_of_positive + no_of_negative );
        roi = Scalar(-1.0);
        
        return labels;
    }
    
    
    CvSVM CustomHOG::trainSVM( Mat & training_data, int no_of_positives, int no_of_negatives, string output_file ) {
        Mat labels = createLabelsMatrix( no_of_positives, no_of_negatives );
        
        CvSVM svm;
        CvSVMParams params;
        params.svm_type     = CvSVM::C_SVC;
        params.kernel_type  = CvSVM::LINEAR;
        params.term_crit    = cvTermCriteria( CV_TERMCRIT_ITER, 100, 1e-6 );
        
        cout << "Start training " << endl;
        svm.train( training_data, labels, Mat(), Mat(), params );
        
        
        cout << "Saving trained SVM " << endl;
        svm.save( output_file.c_str() );
        
        return svm;
    }
    
    Mat CustomHOG::loadFeaturesFromFile( string input_path, int no_of_positive, int no_of_negative ) {
        ifstream input_file( (input_path + "/positive_features.txt" ).c_str() );
        
        vector< vector<float> > positive_features;
        vector< vector<float> > negative_features;
        int minFeaturesSize = 0x7fffffff;
        int index = 0;
        string line;
        
        if( input_file.is_open() ) {
            while( input_file.good() ) {
                index++;
                cout << "reading line: " << index << endl;
                
                getline( input_file, line );
                
                if( !line.empty() ) {
                    vector<float> features;
                    
                    istringstream stream( line );
                    copy( istream_iterator<float>(stream), istream_iterator<float>(), back_inserter(features) );
                    
                    if( minFeaturesSize > (int) features.size() )
                        minFeaturesSize = (int) features.size();
                    
                    positive_features.push_back( features );
                }
                
                if( index == no_of_positive )
                    break;
            }
        }
        
        input_file.close();
        
        input_file.open( (input_path + "/negative_features.txt" ).c_str() );
        index = 0;
        if( input_file.is_open() ) {
            while( input_file.good() ) {
                index++;
                cout << "reading line: " << index << endl;
                
                getline( input_file, line );
                
                if( !line.empty() ) {
                    vector<float> features;
                    
                    istringstream stream( line );
                    copy( istream_iterator<float>(stream), istream_iterator<float>(), back_inserter(features) );
                    
                    if( minFeaturesSize > (int) features.size() )
                        minFeaturesSize = (int) features.size();
                    
                    negative_features.push_back( features );
                }
                
                if( index == no_of_negative )
                    break;
            }
        }
        
        
        input_file.close();
        
        return convertToTrainingDataMatrix( minFeaturesSize, positive_features, negative_features );
    }
    
    void CustomHOG::extractAndSaveHOGFeatures( string positive_files_path, string negative_files_path, string output_dir , int positive_samples, int negative_samples ) {
        path pos_dir_path( positive_files_path );
        vector<string> pos_file_list;
        
        for( directory_iterator dir_itr( pos_dir_path ), dir_end; dir_itr != dir_end; dir_itr++ )
            pos_file_list.push_back( dir_itr->path().string() );
        
        extractAndSaveIndividualHOGFeatures( output_dir + "/positive_features.txt", pos_file_list, positive_samples );
        pos_file_list.clear();
        
        path neg_dir_path( negative_files_path );
        vector<string> neg_file_list;
        
        for( directory_iterator dir_itr( neg_dir_path ), dir_end; dir_itr != dir_end; dir_itr++ )
            neg_file_list.push_back( dir_itr->path().string() );
        
        extractAndSaveIndividualHOGFeatures( output_dir + "/negative_features.txt", neg_file_list, negative_samples );
        neg_file_list.clear();
    }
    
    void CustomHOG::extractAndSaveIndividualHOGFeatures( string output_file_path, vector<string> & file_list, int max_samples ) {
        ofstream output_file( output_file_path.c_str(), ofstream::out );
        ostream_iterator<float> output_iterator( output_file, " " );
        
        if( max_samples == 0 )
            max_samples = (int) file_list.size();
        
        Mat image;
        vector<float> features;
        
        for( int y = 0; y < max_samples; y++ ) {
            cout << "Processing " << file_list[y] << " (" << (y+1) << " / " << max_samples << ")" << endl;
            
            image = imread( file_list[y] );
            features = (vector<float>)this->compute( image );
            
            copy( features.begin(), features.end(), output_iterator );
            output_file << endl;
            
            image.release();
            features.clear();
        }
        
        output_file.close();
    }
    
    
    Mat CustomHOG::compute( Mat& img ) {
        Mat grayscale           = preprocessImage( img );
        vector<Mat> derivatives = applyEdgeFilter( grayscale );
        
        Mat magnitude           = calculateMagnitude( derivatives[0], derivatives[1] );
        Mat angle               = calculateAngle( derivatives[0], derivatives[1] );
        vector<Mat> integrals   = createIntegralBins( magnitude, angle );
        
        Mat feature_vector      = calculateHOGWindow( Rect(0, 0, windowSize.width, windowSize.height), integrals );
        
        return feature_vector;
    }
    
    Mat CustomHOG::preprocessImage( Mat& img ) {
        Mat output;
        cvtColor( img, output, CV_BGR2GRAY );
        
        int col_factor = floor( img.cols / windowSize.width );
        int row_factor = floor( img.rows / windowSize.height );
        int factor = MAX( MIN( col_factor , row_factor ), 1 );
        
        int col_padding = (img.cols - windowSize.width  * factor) / 2;
        int row_padding = (img.rows - windowSize.height * factor) / 2;
        
        Mat( output, Rect( col_padding, row_padding, windowSize.width * factor, windowSize.height * factor ) ).copyTo( output );
                
        if( factor > 1 )
            resize ( output, output, windowSize );
        
        output.convertTo( output, CV_32FC1 );
        return output;
    }
    
    Rect CustomHOG::mergeRects( vector<Rect> rects ) {
        Point2d left_point (999999, 999999);
        Point2d right_point(-1, -1);
        
        for( int i = 0; i < rects.size(); i++ ) {
            if( rects[i].x < left_point.x )
                left_point.x = rects[i].x;
            
            if( rects[i].y < left_point.y )
                left_point.y = rects[i].y;
            
            if( rects[i].x + rects[i].width > right_point.x )
                right_point.x = rects[i].x + rects[i].width;
            
            if( rects[i].y + rects[i].height > right_point.y )
                right_point.y = rects[i].y + rects[i].height;
        }
        
        return Rect( left_point.x, left_point.y, right_point.x - left_point.x, right_point.y - left_point.y );
    }
    
    
    
    Mat CustomHOG::compute( Mat& img, Rect window_size ) {
        Mat grayscale           = preprocessImage( img );
        vector<Mat> derivatives = applyEdgeFilter( grayscale );
        
        Mat magnitude           = calculateMagnitude( derivatives[0], derivatives[1] );
        Mat angle               = calculateAngle( derivatives[0], derivatives[1] );
        vector<Mat> integrals   = createIntegralBins( magnitude, angle );
        
        Mat feature_vector      = calculateHOGWindow( window_size, integrals );
        
        return feature_vector;
    }
    
    vector<Mat> CustomHOG::applyEdgeFilter( Mat& img ) {
        vector<Mat> result(2);
        
        Mat output( abs(img.rows - dxKernel.rows) + 1, abs(img.cols - dxKernel.cols) + 1, img.type() );
        Size dft_size;
        dft_size.width  = getOptimalDFTSize( img.cols + dxKernel.cols - 1 );
        dft_size.height = getOptimalDFTSize( img.rows + dxKernel.rows - 1 );
        
        /* Apply Discrete Fourier Transform (DFT) to image */
        Mat dft_img( dft_size, img.type(), Scalar::all( 0 ) );
        Mat roi_img( dft_img, Rect( 0, 0, img.cols, img.rows ) );
        img.copyTo( roi_img );
        dft( dft_img, dft_img, 0, img.rows );
        
        /* Apply DFT to the dx kernel */
        Mat dft_kernel = Mat::zeros( dft_size, CV_32FC1 );
        Mat roi_kernel = Mat( dft_kernel, Rect(0, 0, dxKernel.cols, dxKernel.rows ) );
        dxKernel.copyTo( roi_kernel );
        
        dft( dft_kernel, dft_kernel, 0, dxKernel.rows );
        
        /* Multiply dft of dx kernel with dft of image */
        mulSpectrums( dft_img, dft_kernel, dft_kernel, DFT_ROWS );
        dft( dft_kernel, dft_kernel, DFT_INVERSE + DFT_SCALE, output.rows );
        dft_kernel( Rect(0, 0, output.cols, output.rows) ).copyTo( output );
        
        //convertScaleAbs( output, result[0] );
        result[0] = output.clone();
        
        /* Apply DFT to the dx kernel */
        dft_kernel = Mat::zeros( dft_size, CV_32FC1 );
        roi_kernel = Mat( dft_kernel, Rect(0, 0, dyKernel.cols, dyKernel.rows ) );
        dyKernel.copyTo( roi_kernel );
        
        dft( dft_kernel, dft_kernel, 0, dyKernel.rows );
        
        /* Multiply dft of dx kernel with dft of image */
        mulSpectrums( dft_img, dft_kernel, dft_kernel, DFT_ROWS );
        dft( dft_kernel, dft_kernel, DFT_INVERSE + DFT_SCALE, output.rows );
        dft_kernel( Rect(0, 0, output.cols, output.rows) ).copyTo( output );
        
        //convertScaleAbs( output, result[1] );
        result[1] = output.clone();
        
        return result;
    }
    
    
    Mat CustomHOG::calculateMagnitude( Mat &dx, Mat &dy ) {
        Mat magnitude;
        sqrt( (dx.mul(dx) + dy.mul(dy)), magnitude );
        return magnitude;
    }
    
    Mat CustomHOG::calculateAngle( Mat &dx, Mat &dy ) {
        /* unnecessary optimization (?) to remove multiplications in loops */
        /* original = dx.step1() * 4 */
        size_t kernel_step = dx.step1() << 2;
        
        /* Preparation to avoid division by zero */
        float * ptr;
        for( int j = 0; j < dx.rows; j++ ) {
            ptr = (float*) ( dx.data + j * kernel_step );
            for( int i = 0; i < dx.cols; i++ ) {
                if( ptr[i] == 0.0f )
                    ptr[i] = 0.00001f;
            }
        }
        
        float rad2deg_multipler = 180.0 / M_PI;
        
        /* Convert to angle in degrees, ranging from 0 to 180 degrees */
        Mat angle = dy / dx;
        for( int j = 0; j < angle.rows; j++ ) {
            ptr = (float*) ( angle.data + j * kernel_step );
            for( int i = 0; i < angle.cols; i++ ) {
                ptr[i] = atan( ptr[i] ) * rad2deg_multipler + 90.0 ;
            }
        }
        return angle;
    }
    
    
    vector<Mat> CustomHOG::createIntegralBins( Mat &magnitude, Mat &angle ) {
        vector<Mat> bins(noOfBins);
        
        for( int i = 0; i < noOfBins; i++ )
            bins[i] = Mat( angle.rows + 1, angle.cols + 1, CV_32FC1, Scalar::all(0) );
        
        size_t step     = angle.step1() << 2;
        size_t bin_step = bins[0].step1() << 2;
        
        float * bin_ptrs[9];
        float * magnitude_ptr;
        float * angle_ptr;
        
        for( int j = 0; j < angle.rows; j++ ) {
            /* Setting up the pointers to the matrices */
            angle_ptr       = (float*) ( angle.data + j * step );
            magnitude_ptr   = (float*) ( magnitude.data + j * step );
            
            for( int i = 0; i < noOfBins; i++ )
                bin_ptrs[i] = (float*) ( bins[i].data + j * bin_step );
            
            for( int i = 0; i < angle.cols; i++ ) {
                
                if( angle_ptr[i] <= 20.0f )
                    bin_ptrs[0][i] = magnitude_ptr[i];
                else if( angle_ptr[i] <= 40.0f )
                    bin_ptrs[1][i] = magnitude_ptr[i];
                else if( angle_ptr[i] <= 60.0f )
                    bin_ptrs[2][i] = magnitude_ptr[i];
                else if( angle_ptr[i] <= 80.0f )
                    bin_ptrs[3][i] = magnitude_ptr[i];
                else if( angle_ptr[i] <= 100.0f )
                    bin_ptrs[4][i] = magnitude_ptr[i];
                else if( angle_ptr[i] <= 120.0f )
                    bin_ptrs[5][i] = magnitude_ptr[i];
                else if( angle_ptr[i] <= 140.0f )
                    bin_ptrs[6][i] = magnitude_ptr[i];
                else if( angle_ptr[i] <= 160.0f )
                    bin_ptrs[7][i] = magnitude_ptr[i];
                else if( angle_ptr[i] <= 180.0f )
                    bin_ptrs[8][i] = magnitude_ptr[i];
            }
        }
        
        for( int i = 0; i < noOfBins; i++ )
            integral( bins[i], bins[i] );
        
        return bins;
    }
    
    
    Mat CustomHOG::calculateHOGWindow( Rect window, vector<Mat> integrals ) {
        Mat feature_vector( 1, totalBlocks * featureBlockLen, CV_32FC1 );
        Mat vector_block;
        int start_col = 0;
        
        int total_block_height = cellSize.height * blockSize.height;
        int total_block_width  = cellSize.width * blockSize.width;
        
        for( int y = window.y; y <= window.y + window.height - total_block_height; y += cellSize.height * stepOverlap ) {
            for( int x = window.x; x <= window.x + window.width - total_block_width; x += cellSize.width * stepOverlap ) {
                
                vector_block = feature_vector.colRange( start_col, start_col + featureBlockLen );
                calculateHOBlock( Rect( x, y, total_block_width, total_block_height ), vector_block, integrals );
                start_col += featureBlockLen;
            }
        }
        
        return feature_vector;
    }
    
    void CustomHOG::calculateHOBlock( Rect block, Mat &hog_block, vector<Mat> &integrals ) {
        Mat cell;
        int start_col = 0;
        
        for( int y = block.y; y <= block.y + block.height - cellSize.height; y+= cellSize.height ) {
            for( int x = block.x; x <= block.x + block.width - cellSize.width; x+= cellSize.width ) {
                cell = hog_block.colRange( start_col, start_col + noOfBins );
                calculateHOGRect( Rect( x, y, cellSize.width, cellSize.height ), cell, integrals );
                start_col += noOfBins;
            }
        }

        if( normalization != -1 )
            cv::normalize( hog_block, hog_block, 1, 0, normalization );
    }
    
    void CustomHOG::calculateHOGRect( Rect cell, Mat &hog_cell, vector<Mat> &integrals ) {
        float * ptr = hog_cell.ptr<float>();
        
        size_t step = integrals[0].step1() << 2;
        size_t y1_step = cell.y * step;
        size_t y2_step = (cell.y + cell.height) * step;
        
        for( int i = 0; i < noOfBins; i++ ) {
            float a = ((double*)( integrals[i].data + y1_step ))[cell.x];
            float b = ((double*)( integrals[i].data + y2_step ))[cell.x + cell.width];
            float c = ((double*)( integrals[i].data + y1_step ))[cell.x + cell.width];
            float d = ((double*)( integrals[i].data + y2_step ))[cell.x];
            ptr[i] = a + b - c - d;
        }
        
        //if( normalization != -1 )
        //    cv::normalize( hog_cell, hog_cell, 1, 0, normalization );
    }
}