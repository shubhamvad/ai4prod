/*

GNU GPL V3 License

Copyright (c) 2020 Eric Tondelli. All rights reserved.

This file is part of Ai4prod.

Ai4prod is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Ai4prod is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Ai4prod.  If not, see <http://www.gnu.org/licenses/>

*/

#include "catch.hpp"

#include <iostream>

#include "torch/torch.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "instancesegmentation.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// include namespace
using namespace ai4prod;

using namespace instanceSegmentation;
using namespace cv;
using namespace std::chrono;
using namespace torch::indexing;

//this is needed if you want to scan and entire folder
// img1.jpg
// img2.jpg
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;


#if defined AICPU || defined TENSORRT
TEST_CASE("Init Instance Segmentation Yolact Cpu")
{
    std::cout << "TEST INIT YOLACT ON CPU" << std::endl;

    std::unique_ptr<Yolact> yolact = std::make_unique<Yolact>();

    bool initFlag = false;

    initFlag = yolact->init("../Model/yolact.onnx", 550, 550, 80, Cpu, "../test/yolactCPU");
    REQUIRE(initFlag == true);
}

TEST_CASE("Test Instance Segmentation Output Yolact CPU")
{
    std::cout << "TEST INSTANCE SEGMENTATION OUTPUT Yolact Cpu" << std::endl;

    std::vector<std::vector<int>> gtBbox={{105,21,140,120},{57,50,22,53},{69,57,7,6}};

    std::unique_ptr<Yolact> yolact = std::make_unique<Yolact>();

    yolact->init("../Model/yolact.onnx", 550, 550, 80, Cpu, "../test/yolactCPU");
    Mat img;
    Mat maskOriginal= imread("../Images/instanceSegmentation/maskFull.png",0);

    img = imread("../Images/instanceSegmentation/yolact-federer.jpeg");
    yolact->preprocessing(img);
    yolact->runmodel();

    auto result = yolact->postprocessing();

    auto resultBbox=yolact->getCorrectBbox(result);

    for(int i=0;i<resultBbox.size();i++){

		REQUIRE(resultBbox[i].x==gtBbox[i][0]);
        REQUIRE(resultBbox[i].y==gtBbox[i][1]);
        REQUIRE(resultBbox[i].height==gtBbox[i][2]);
        REQUIRE(resultBbox[i].width==gtBbox[i][3]);
 	}


    std::cout <<"Check MASK" <<std::endl;


    //check Mask
    auto resultMask = yolact->getCorrectMask(result);
    
    cv::Mat resultMaskFull= resultMask[0] | resultMask[1] | resultMask[2];

    
    std::cout <<maskOriginal.type() <<std::endl;
    std::cout <<resultMaskFull.type() <<std::endl;
    Mat diffMask= maskOriginal - resultMaskFull;
    

    //count the number of black pixel
    int count_black = cv::countNonZero(diffMask == 0);
    REQUIRE(count_black==50268);
    
    

}

#endif

#if defined(TENSORRT)

TEST_CASE("Init Instance Segmentation Yolact Tensorrt")
{
    std::cout << "TEST INIT YOLACT ON Tensorrt" << std::endl;

    std::unique_ptr<Yolact> yolact = std::make_unique<Yolact>();

    bool initFlag = false;

    initFlag = yolact->init("../Model/yolact.onnx", 550, 550, 80, TensorRT, "../test/yolactTensorrt");
    REQUIRE(initFlag == true);

    yolact.release();
    cudaDeviceReset();
}





TEST_CASE("Test Instance Segmentation Output Yolact Tensorrt")
{
    std::cout << "TEST INSTANCE SEGMENTATION OUTPUT Yolact TENSORRT" << std::endl;

    std::vector<std::vector<int>> gtBbox={{105,21,140,120},{56,50,21,55},{68,57,7,7}};

    std::unique_ptr<Yolact> yolact = std::make_unique<Yolact>();

    yolact->init("../Model/yolact.onnx", 550, 550, 80, TensorRT, "../test/yolactTensorrt");
    Mat img;
    Mat maskOriginal= imread("../Images/instanceSegmentation/maskFull.png",0);

    img = imread("../Images/instanceSegmentation/yolact-federer.jpeg");
    yolact->preprocessing(img);
    yolact->runmodel();

    auto result = yolact->postprocessing();

    auto resultBbox=yolact->getCorrectBbox(result);

    for(int i=0;i<resultBbox.size();i++){

		REQUIRE(resultBbox[i].x==gtBbox[i][0]);
        REQUIRE(resultBbox[i].y==gtBbox[i][1]);
        REQUIRE(resultBbox[i].height==gtBbox[i][2]);
        REQUIRE(resultBbox[i].width==gtBbox[i][3]);
 	}


    std::cout <<"Check MASK" <<std::endl;


    //check Mask
    auto resultMask = yolact->getCorrectMask(result);
    
    cv::Mat resultMaskFull= resultMask[0] | resultMask[1] | resultMask[2];

    
    std::cout <<maskOriginal.type() <<std::endl;
    std::cout <<resultMaskFull.type() <<std::endl;
    Mat diffMask= maskOriginal - resultMaskFull;
    

    //count the number of black pixel
    int count_black = cv::countNonZero(diffMask == 0);
    REQUIRE(count_black==49970);
    
    yolact.release();
    cudaDeviceReset();

}

#endif