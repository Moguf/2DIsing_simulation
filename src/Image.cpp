#include <iostream>
#include <stdio.h>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <opencv2/opencv.hpp>
#include <iostream>

#include "Ising2D.hpp"
#include "Image.hpp"

void Image::draw(SPIN *hS,char *filename){

    cv::Mat png(ROW,COL,CV_16UC1);
    cv::Mat img;
    cv::Vec3b *src;

    for(int i = 0 ; i < ROW-1 ; i++){
        src = png.ptr<cv::Vec3b>(i);

        for(int j = 0 ; j < COL-1 ; j++){
            if(hS[i+COL*j] == 1){
                src[j][0] = (unsigned char)0;
                src[j][1] = (unsigned char)0;
                src[j][2] = (unsigned char)0;
            }

            else{
                src[j][0] = (unsigned char)-1;
                src[j][1] = (unsigned char)-1;
                src[j][2] = (unsigned char)-1;
            }

        }

    }
    png.convertTo(img,CV_16U);
    cv::imwrite(filename,img);
}

void Image::printCVversion(){
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    std::cout << "Major version : " << CV_MAJOR_VERSION << std::endl;
    std::cout << "Minor version : " << CV_MINOR_VERSION << std::endl;
    std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << std::endl;
}
