#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <opencv2/opencv.hpp>
#include <iostream>

#include "Image.hpp"
#include "Ising2D.hpp"


void Image::draw(){
    cv::Mat png(ROW,COL,CV_16UC1);
    
    cv::imwrite("test.png",png);            
    cv::Vec3b *src = png.ptr<cv::Vec3b>(0);
}

