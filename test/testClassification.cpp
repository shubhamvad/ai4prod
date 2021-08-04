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

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "classification.h"
using namespace ai4prod;
using namespace classification;
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace cv;

namespace fs = std::experimental::filesystem;

#if defined(AICPU) || defined(TENSORRT)

TEST_CASE("Init classification Resnet Cpu")
{   
    std::cout << "TEST INIT RESNET ON CPU"<<std::endl;

    ResNet50 resnet;
    bool initFlag = false;
    initFlag = resnet.init("../Model/resnet50.onnx", 256, 256, 1000, 5, Cpu,
                           "../test/testResnetCpu");

    REQUIRE(initFlag == true);
}

TEST_CASE("Test Classification Output Resnet Cpu")
{
    std::cout << "TEST CLASSIFICATION OUTPUT RESNET ON CPU"<<std::endl;
    ResNet50 resnet;
    bool initFlag = false;
    initFlag = resnet.init("../Model/resnet50.onnx", 256, 256, 1000, 5, Cpu,
                           "../test/testResnetCpu");

    string image_id = "../Images/classification/dog.jpeg";

    Mat img;

    //read image with opencv
    img = imread(image_id.c_str());

    resnet.preprocessing(img);

    //run model on img
    resnet.runmodel();

    std::tuple<torch::Tensor, torch::Tensor> prediction = resnet.postprocessing();

    REQUIRE(std::get<0>(prediction)[0].item<float>() == 208);
}

#endif

#if defined(TENSORRT)

TEST_CASE("Init classification Resnet Tensorrt")
{   
    std::cout << "TEST INIT RESNET TENSORRT "<<std::endl;

    

    std::unique_ptr<ResNet50> resnet= make_unique<ResNet50>();
    
    bool initFlag = false;
    initFlag = resnet->init("../Model/resnet50.onnx", 256, 256, 1000, 5, TensorRT,
                           "../test/testResnetTensorrt");

    fs::path p1{"../test/testResnetTensorrt"};
    int count{};

    for (auto &p : fs::directory_iterator(p1))
    {
        ++count;
    }

    REQUIRE(initFlag == true);
    //check if tensorrtModel is saved
    REQUIRE(count > 1);

    resnet.reset();
    cudaDeviceReset();
}


TEST_CASE("Test Classification Output Resnet TensorRT")
{
    std::cout << "TEST CLASSIFICATION OUTPUT RESNET ON TENSORRT"<<std::endl;
    std::unique_ptr<ResNet50> resnet= make_unique<ResNet50>();
    bool initFlag = false;
    initFlag = resnet->init("../Model/resnet50.onnx", 256, 256, 1000, 5, TensorRT,
                           "../test/testResnetTensorrt");

    string image_id = "../Images/classification/dog.jpeg";

    Mat img;

    //read image with opencv
    img = imread(image_id.c_str());

    resnet->preprocessing(img);

    //run model on img
    resnet->runmodel();

    std::tuple<torch::Tensor, torch::Tensor> prediction = resnet->postprocessing();

    REQUIRE(std::get<0>(prediction)[0].item<float>() == 208);

    resnet.reset();
    cudaDeviceReset();
}

#endif