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

TEST_CASE("Init Instance Segmentation Yolact Cpu")
{
    std::cout << "TEST INIT YOLACT ON CPU" << std::endl;

   std::unique_ptr<Yolact> yolact = std::make_unique<Yolact>();


    bool initFlag = false;

    initFlag = yolact->init("../Model/yolact.onnx", 550, 550, 80, Cpu, "../test/yolactCPU");
    REQUIRE(initFlag == true);
}

TEST_CASE("Test Object Detection Output Yolact Tensorrt")
{
    std::cout << "TEST OBJECT DETECTION OUTPUT Yolact TENSORRT" << std::endl;

    std::unique_ptr<Yolact> yolact = std::make_unique<Yolact>();

    yolact->init("../Model/yolact.onnx", 550, 550, 80, Cpu, "../test/yolactCPU");
    Mat img;
	img= imread("../yolact-federer.jpeg");	

    

}
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

// TEST_CASE("Test Object Detection Output Yolact Cpu")
// {
//     std::cout << "TEST OBJECT DETECTION OUTPUT YOLACT CPU" << std::endl;

//     Hrnet hrnet;
//     bool initFlag = false;
//     initFlag = hrnet.init("../Model/hrnet.onnx", 256, 192, 80, Cpu,
//                           "../test/testHrnetCpu");

//     REQUIRE(initFlag == true);
// }