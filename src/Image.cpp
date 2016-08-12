#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <opencv2/opencv.hpp>
#include <iostream>

#include "Ising2D.hpp"
#include "Image.hpp"

void Image::draw(SPIN *hS){
    cv::Mat png(ROW,COL,CV_16UC1);
    cv::Mat img;
    cv::Vec3b *src;
    for(int i = 0 ; i < 1024 ; i++){
        src = png.ptr<cv::Vec3b>(i);
        for(int j = 0 ; j < COL ; j++){
            if(hS[i+COL*j] == 1){
                src[j][0] = (unsigned char)0;
                src[j][1] = (unsigned char)0;
                src[j][2] = (unsigned char)0;
            }else{
                src[j][0] = (unsigned char)-1;
                src[j][1] = (unsigned char)-1;
                src[j][2] = (unsigned char)-1;
            }
        }
    }
    png.convertTo(img,CV_16U);
    cv::imwrite("test.png",img);
    std::cout<< "Before ending."<<std::endl;
}

