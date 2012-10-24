//
//  MeanShift.cpp
//  OpenCVTutorial
//
//  Created by Saburo Okita on 23/10/12.
//  Copyright (c) 2012 Saburo Okita. All rights reserved.
//

#include "MeanShift.h"

MeanShift::MeanShift( vector<CustomPoint> &points, double sigma_x, double sigma_y, double sigma_s, int point_index, int max_iteration ){
    this->points = points;
    this->sigmaX = sigma_x;
    this->sigmaY = sigma_y;
    this->sigmaS = sigma_s;
    this->pointIndex = point_index;
    this->maxIteration = max_iteration;
    
}

CustomPoint MeanShift::computeMode( CustomPoint point ){
    Mat hh = computeMatrixHH( point );
    double fact[3] = {0.0, 0.0, 0.0 };
    
    for( int i = 0; i < points.size(); i++ ) {
        double weight = computeWeight( i,  point );
        Mat hi = computeMatrixH( points[i].scale );
        
        hi.at<float>(0, 0) = 1 / hi.at<float>(0, 0);
        hi.at<float>(1, 1) = 1 / hi.at<float>(1, 1);
        hi.at<float>(2, 2) = 1 / hi.at<float>(2, 2);
        
        fact[0] += weight * hi.at<float>(0, 0) * points[i].x;
        fact[1] += weight * hi.at<float>(1, 1) * points[i].y;
        fact[2] += weight * hi.at<float>(2, 2) * points[i].scale;
    }
    
    int x = (int) (hh.at<float>(0, 0) * fact[0] );
    int y = (int) (hh.at<float>(1, 1) * fact[1] );
    double scale = (hh.at<float>(2, 2) * fact[2] );
    
    return CustomPoint( x, y, scale, point.weight );
}

Mat MeanShift::computeMatrixHH( CustomPoint y ){
    Mat hh = Mat::zeros(3, 3, CV_32FC1); 
    
    for( int i = 0; i < points.size(); i++ ) {
        Mat h = computeMatrixH( points[i].scale );
        h.at<float>(0, 0) = 1 / h.at<float>(0, 0);
        h.at<float>(1, 1) = 1 / h.at<float>(1, 1);
        h.at<float>(2, 2) = 1 / h.at<float>(2, 2);
        
        hh.at<float>(0, 0) += computeWeight( i, y ) * h.at<float>(0, 0);
        hh.at<float>(1, 1) += computeWeight( i, y ) * h.at<float>(1, 1);
        hh.at<float>(2, 2) += computeWeight( i, y ) * h.at<float>(2, 2);
    }
    
    hh.at<float>(0, 0) = 1 / hh.at<float>(0, 0);
    hh.at<float>(1, 1) = 1 / hh.at<float>(1, 1);
    hh.at<float>(2, 2) = 1 / hh.at<float>(2, 2);
    
    
    return hh;
}

Mat MeanShift::computeMatrixH( double scale ){
    Mat h = Mat::zeros(3, 3, CV_32FC1); 
    
    h.at<float>(0, 0) = pow( exp( scale ) * sigmaX, 2 );
    h.at<float>(1, 1) = pow( exp( scale ) * sigmaY, 2 );
    h.at<float>(2, 2) = pow( sigmaS, 2 );
    
    return h;
}

double MeanShift::computeWeight( int i, CustomPoint point){
    double scale = points[i].scale;
    Mat Hi = computeMatrixH( scale );
    double contribution = contributePoint( point, points[i], Hi );
    double den = 0.0f;
    
    for( int j = 0; j< points.size(); j++ ) {
        Mat Hj = computeMatrixH( points[j].scale );
        den += contributePoint( point, points[j], Hj );
    }
    
    return contribution / den;
}

double MeanShift::computeHNorm( Mat& H ){
    return MAX( MAX( abs(H.at<float>(0, 0)), abs(H.at<float>(1, 1)) ), abs(H.at<float>(2, 2)) );
}

double MeanShift::computeMahalanaboisDistance( CustomPoint y, CustomPoint yi, Mat &Hi ){
    double distance = 0.0;
    distance += pow( y.x - yi.x, 2.0 ) / Hi.at<float>(0, 0);
    distance += pow( y.y - yi.y, 2.0 ) / Hi.at<float>(1, 1);
    distance += pow( (double) (y.scale - yi.scale), 2.0 ) / Hi.at<float>(2, 2);
    return distance;
}

double MeanShift::contributePoint( CustomPoint y, CustomPoint yi, Mat &Hi ){
    double h_norm = computeHNorm( Hi );
    double mahalanabois_distance = computeMahalanaboisDistance( y, yi, Hi );
    return pow( h_norm, -0.5)
            * weightTransformation( yi.weight, 0 )
            * exp( -mahalanabois_distance/2.0 ) ;
}

double MeanShift::weightTransformation( double weight, int c ){
    if( weight < c )
        return 0.0;
    return weight - c;
}


void MeanShift::calc( vector<CustomPoint> &modes ){
    cout << "Running meanshift for " << pointIndex << endl;
    
    CustomPoint old;
    CustomPoint point = points[pointIndex];
    for( int i = 0; i < maxIteration; i++ ) {
        old = point;
        point = computeMode( old );
        
        if( old.equals( point ))
            break;
    }
    
    if( point.existsIn(modes) == false )
        modes.push_back( point );
}