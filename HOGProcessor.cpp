//
//  HOGProcessor.cpp
//  OpenCVTutorial
//
//  Created by Saburo Okita on 28/8/12.
//  Copyright (c) 2012 Saburo Okita. All rights reserved.
//

#include "HOGProcessor.h"
#include "ImageProcessor.h"
#include "Utils.h"
#include "Stopwatch.h"

HOGProcessor::HOGProcessor() {
    cellSize    = Size(4, 4);
    blockSize   = Size( 2, 2 );
    stepOverlap = 0.5f;
}

HOGProcessor::~HOGProcessor() {
    
}

void HOGProcessor::setParams( Size cell_size, Size block_size, float step_overlap ) {
    this->cellSize = cell_size;
    this->blockSize = block_size;
    this->stepOverlap = step_overlap;
}

Mat HOGProcessor::train64x128( const char * filepath, Size window, int no_of_samples, const char * save_filename, int normalization ) {
    
    int block_width     = blockSize.width;
    int block_height    = blockSize.height;
    int cell_width      = cellSize.width;
    int cell_height     = cellSize.height;
    
    int feature_block_len   = 9 * block_height * block_width;
    int width_overlaps      = ((window.width  - cell_width  * block_width)  / (cell_width  * stepOverlap)) + 1;
    int height_overlaps     = ((window.height - cell_height * block_height) / (cell_height * stepOverlap)) + 1;
    int total_blocks        = width_overlaps * height_overlaps;
    
    Mat training_matrix( no_of_samples, total_blocks * feature_block_len, CV_32FC1 );
    Mat row_matrix;
    Mat image_feature_vector;
    vector<Mat> integrals;

    cout << "Beginning to extract HOG Features" << endl;
    
    int j = 0;
    vector<path> files = Utils::listFiles( filepath );
    for( int i = 0; i < files.size(); i++ ) {
        cout << "Extracting features for " << files[i].string() << endl;
        Mat image = imread( files[i].string() );
        Mat resized;
        resize( image, resized, window );
        
        integrals = calculateIntegralHOG( resized );
             
        
        row_matrix = training_matrix.rowRange( j, j + 1);
        
        image_feature_vector = calculateHOGWindow( integrals, Rect(0, 0, window.width, window.height), normalization );
        image_feature_vector.copyTo( row_matrix );
        
        j++;
        if( j == no_of_samples )
            break;
    }
    
    if( save_filename != NULL ) {
        FileStorage file_storage( save_filename, FileStorage::WRITE );
        file_storage << "training_matrix" << training_matrix;
    }
        
    
    return training_matrix;
}

Mat HOGProcessor::calculateHOGWindow( vector<Mat> integrals, Rect window, int normalization ) {
    cout << "[START] calculateHOGWindow" << endl;
    
    int block_x, block_y;
    int cell_width      = cellSize.width;
    int cell_height     = cellSize.height;
    int block_width     = blockSize.width;
    int block_height    = blockSize.height;
    
    int feature_block_len   = 9 * block_height * block_width;
    int width_overlaps      = ((window.width  - cell_width  * block_width)  / (cell_width  * stepOverlap)) + 1;
    int height_overlaps     = ((window.height - cell_height * block_height) / (cell_height * stepOverlap)) + 1;
    int total_blocks        = width_overlaps * height_overlaps;
    
    Mat feature_vector( 1, total_blocks * feature_block_len, CV_32FC1 );
    Mat vector_block;
    int start_col = 0;
    
    for( block_y = window.y;
         block_y <= window.y + window.height - cell_height * block_height;
         block_y += cell_height * stepOverlap ) {
        
        for( block_x = window.x;
             block_x <= window.x + window.width - cell_width * block_width;
             block_x += cell_width * stepOverlap ) {
            
            vector_block = feature_vector.colRange( start_col, start_col + 36 );
            
            calculateHOGBlock( Rect(block_x, block_y, cell_width * block_width, cell_height * block_height), vector_block, integrals, cellSize, normalization );
            
            start_col += feature_block_len;
        }
    }
    
    cout << "[END] calculateHOGWindow" << endl;
    
    return feature_vector;
}

Mat HOGProcessor::calculateHOGWindow( Mat& image, vector<Mat> integrals, Rect window, int normalization ) {
    cout << "[START] calculateHOGWindow" << endl;
    
    int block_x, block_y;
    int cell_width      = cellSize.width;
    int cell_height     = cellSize.height;
    int block_width     = blockSize.width;
    int block_height    = blockSize.height;
    
    int feature_block_len   = 9 * block_height * block_width;
    int width_overlaps      = ((window.width - cell_width * block_width) / (cell_width * stepOverlap) ) + 1;
    int height_overlaps     = ((window.height - cell_height * block_height) / (cell_height * stepOverlap) ) + 1;
    int total_blocks        = width_overlaps * height_overlaps;
    
    Mat feature_vector( 1, total_blocks * feature_block_len, CV_32FC1 );
    Mat vector_block;
    int start_col = 0;

    Mat hog_show(9, 9, image.type(), Scalar(0.0));
    
    for( block_y = window.y;
         block_y <= window.y + window.height - cell_height * block_height ;
         block_y += cell_height * stepOverlap ) {
         
        for( block_x = window.x;
            block_x <= window.x + window.width - cell_width * block_width ;
            block_x += cell_width * stepOverlap ) {
            vector_block = feature_vector.colRange( start_col, start_col + 36 );
    
            calculateHOGBlock( Rect(block_x, block_y, cell_width * block_width, cell_height * block_height), vector_block, integrals, Size( cell_width, cell_height ), normalization );
         
            // feature_block_len = 36
            start_col +=  feature_block_len;
            
            for( int row = 0; row < 9; row++ ) {
                for( int col = 0; col < 9; col++ ) {
                    float cell = vector_block.at<float>( 0, 9 * row + col );
                    hog_show.at<uchar>(row, col) = cell * 255;
                }
            }
            
            imshow( "", hog_show );
            waitKey( 0 );
        }
    }

    cout << "[END] calculateHOGWindow" << endl;
    return feature_vector;
}

vector<Mat> HOGProcessor::calculateIntegralHOG( Mat& image ) {
    cout << "[START] calculateIntegralHOG" << endl;
    
    Mat grayscale;
    cvtColor( image, grayscale, CV_RGB2GRAY );
    equalizeHist( grayscale, grayscale );
    
    Mat dx = ImageProcessor::applySobel( grayscale, 3, 1, 0);
    Mat dy = ImageProcessor::applySobel( grayscale, 3, 0, 1);
    grayscale.release();

    
    vector<Mat> bins(9);
    for( int i = 0; i < 9; i++ )
        bins[i] = Mat( image.rows, image.cols, CV_32FC1, Scalar(0) );
    
    vector<Mat> integrals(9);
    for( int i = 0; i < 9; i++ )
        integrals[i] = Mat( image.rows + 1, image.cols + 1, CV_64FC1, Scalar(0) );
 
    float dx_value, dy_value, temp_gradient;
    float* pointers[9];
    
    for( int y = 0; y < image.rows; y++ ) {
        float * dx_ptr = (float *)(dx.data + y * dx.step1() );
        float * dy_ptr = (float *)(dy.data + y * dy.step1() );

        for( int i = 0; i < 9; i++ )
            pointers[i] = (float*)(bins[i].ptr() + y * bins[i].step1());
        
        for( int x = 0; x < image.cols; x++ ) {
            dx_value = (*dx_ptr++);
            if( isnan(dx_value ) )
                dx_value = 0.0f;
            
            dy_value = (*dy_ptr++);
            if( isnan(dy_value ) )
                dy_value = 0.0f;
            
            if( dx_value == 0.0f )
                temp_gradient = atan(dy_value / (dx_value + 0.00001)) * (180 / M_PI) + 90;
            else
                temp_gradient = atan(dy_value / (dx_value)) * (180 / M_PI) + 90;
            
            float temp_magnitude = sqrt( dx_value * dx_value + dy_value * dy_value );
            if( isinf( temp_magnitude ) )
                temp_magnitude = 0.0;
            
            if( temp_gradient <= 20.0 )
                pointers[0][x] = temp_magnitude;
            else if( temp_gradient <= 40.0 )
                pointers[1][x] = temp_magnitude;
            else if( temp_gradient <= 60.0 )
                pointers[2][x] = temp_magnitude;
            else if( temp_gradient <= 80.0 )
                pointers[3][x] = temp_magnitude;
            else if( temp_gradient <= 100.0 )
                pointers[4][x] = temp_magnitude;
            else if( temp_gradient <= 120.0 )
                pointers[5][x] = temp_magnitude;
            else if( temp_gradient <= 140.0 )
                pointers[6][x] = temp_magnitude;
            else if( temp_gradient <= 160.0 )
                pointers[7][x] = temp_magnitude;
            else if( temp_gradient <= 180.0 )
                pointers[8][x] = temp_magnitude;
        }
    }
    
    dx.release();
    dy.release();
    
    for( int i = 0; i < 9; i++ ) {
        integral( bins[i], integrals[i] );
        bins[i].release();
        pointers[i] = NULL;
    }
    
    cout << "[END] calculateIntegralHOG" << endl;
    return integrals;
}

Rect HOGProcessor::mergeRects( vector<Rect> rectangles ) {
    cout << "[START] mergeRects" << endl;
    
    int x1 = 999999;
    int y1 = 999999;
    int x2 = -1;
    int y2 = -1;
    
    for( vector<Rect>::iterator itr = rectangles.begin(); itr != rectangles.end(); itr++ ) {
        if( itr->x < x1 )
            x1 = itr->x;
        
        if( itr->y < y1 )
            y1 = itr->y;
        
        if( (itr->x + itr->width) > x2 )
            x2 = itr->x + itr->width;
        
        if( (itr->y + itr->height) > y2 )
            y2 = itr->y + itr->height;
    }
    
    cout << "[END] mergeRects" << endl;
    return Rect( x1, y1, x2 - x1, y2 - y1 );
}

void HOGProcessor::calculateHOGBlock( Rect block, Mat& hog_block, vector<Mat>& integrals, Size cell_size, int normalization ) {
    cout << "[START] calculateHOGBlock" << endl;
    
    int x, y;
    Mat cell;
    int start_col = 0;
    
    for( y = block.y; y <= block.y + block.height - cell_size.height; y += cell_size.height  ) {
        for( x = block.x; x <= block.x + block.width - cell_size.width; x += cell_size.width ) {
            cell = hog_block.colRange( start_col, start_col + 9 );
            calculateHOGRect( Rect(x, y, cell_size.width, cell_size.height), cell, integrals, normalization );
            start_col += 9;
        }
    }
    
    if( normalization != -1 )
        normalize( hog_block, hog_block, 1, 0, normalization );
    
    cout << "[END] calculateHOGBlock" << endl;
}

void HOGProcessor::calculateHOGRect( Rect cell, Mat& hog_cell, vector<Mat>& integrals, int normalization ) {
    cout << "[START] calculateHOGRect" << endl;
    
    float * ptr = hog_cell.ptr<float>();
    
    for( int i = 0; i < 9; i++ ) {
        float a = ((double*)(integrals[i].data + cell.y * integrals[i].step1()))[cell.x];
        float b = ((double*)(integrals[i].data + (cell.y + cell.height) * integrals[i].step1()))[cell.x + cell.width];
        float c = ((double*)(integrals[i].data + cell.y * integrals[i].step1()))[cell.x + cell.width];
        float d = ((double*)(integrals[i].data + (cell.y + cell.height) * integrals[i].step1()))[cell.x];
        
        ptr[i] = (a + b) - (c + d);
    }
    
    if( normalization != -1 )
        cv::normalize( hog_cell, hog_cell, 1, 0, normalization );
    
    cout << "[END] calculateHOGRect" << endl;
}