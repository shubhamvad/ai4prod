
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
#include "objectdetection.h"
using namespace ai4prod;
using namespace objectDetection;
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace cv;

namespace fs = std::experimental::filesystem;
#if defined AICPU || defined TENSORRT
TEST_CASE("Init Object Detection yolov4 Cpu")
{
    std::cout << "TEST INIT YOLOV4 ON CPU" << std::endl;

    Yolov4 yolov4;
    bool initFlag = false;
    initFlag = yolov4.init("../Model/yolov4_608.onnx", 608, 608, 80, Cpu, "../test/testYolov4Cpu");

    REQUIRE(initFlag == true);
}

TEST_CASE("Test Object Detection Output Yolov4 Cpu")
{
    std::cout << "TEST OBJECT DETECTION OUTPUT YOLOV4 CPU" << std::endl;

    Yolov4 yolov4;
    yolov4.init("../Model/yolov4_608.onnx", 608, 608, 80, Cpu, "../test/testYolov4Cpu");

    Mat img;
    std::string image_id = "../Images/objectdetection/dog.jpg";
    img = imread(image_id.c_str());

    yolov4.preprocessing(img);

    yolov4.runmodel();

    torch::Tensor result = yolov4.postprocessing();

    std::vector<std::vector<int>> coordinate{{121, 121, 449, 303, 1},
                                             {466, 77, 214, 92, 7},
                                             {130, 223, 182, 318, 16}};

    int indexObjectThres = 0;

    for (int i = 0; i < result.sizes()[0]; i++)
    {

        int x = result[i][0].item<int>();
        int y = result[i][1].item<int>();
        int width = result[i][2].item<int>();
        int height = result[i][3].item<int>();
        int cls = result[i][5].item<int>();

        if (result[i][4].item<float>() > 0.7)
        {
            std::cout << "x " << x << "y " << y << "width " << width << "height " << height << "prob " << result[i][4].item<float>() << "class " << result[i][5].item<int>() << std::endl;
            std::cout << "ITERATION " << i << std::endl;

            REQUIRE(x == coordinate[indexObjectThres][0]);
            REQUIRE(y == coordinate[indexObjectThres][1]);
            REQUIRE(width == coordinate[indexObjectThres][2]);
            REQUIRE(height == coordinate[indexObjectThres][3]);
            REQUIRE(cls == coordinate[indexObjectThres][4]);
            indexObjectThres = indexObjectThres + 1;
        }
    }
}
#endif


#if defined(TENSORRT)
TEST_CASE("Init Object Detection yolov4 Tensorrt")
{
    std::cout << "TEST INIT YOLOV4 ON TENSORRT" << std::endl;

    std::unique_ptr<Yolov4> yolov4 = std::make_unique<Yolov4>();

    bool initFlag = false;
    initFlag = yolov4->init("../Model/yolov4_608.onnx", 608, 608, 80, TensorRT, "../test/testYolov4TensorRt");

    REQUIRE(initFlag == true);

    yolov4.reset();
    cudaDeviceReset();
}

TEST_CASE("Test Object Detection Output Yolov4 Tensorrt")
{
    std::cout << "TEST OBJECT DETECTION OUTPUT YOLOV4 TENSORRT" << std::endl;

    std::unique_ptr<Yolov4> yolov4 = std::make_unique<Yolov4>();
    yolov4->init("../Model/yolov4_608.onnx", 608, 608, 80, TensorRT, "../test/testYolov4TensorRt");

    Mat img;
    std::string image_id = "../Images/objectdetection/dog.jpg";
    img = imread(image_id.c_str());

    yolov4->preprocessing(img);

    yolov4->runmodel();

    torch::Tensor result = yolov4->postprocessing();

    std::vector<std::vector<int>> coordinate{{121, 121, 449, 303, 1},
                                             {466, 77, 214, 92, 7},
                                             {130, 223, 182, 318, 16}};

    int indexObjectThres = 0;

    for (int i = 0; i < result.sizes()[0]; i++)
    {

        int x = result[i][0].item<int>();
        int y = result[i][1].item<int>();
        int width = result[i][2].item<int>();
        int height = result[i][3].item<int>();
        int cls = result[i][5].item<int>();

        if (result[i][4].item<float>() > 0.7)
        {
            std::cout << "x " << x << "y " << y << "width " << width << "height " << height << "prob " << result[i][4].item<float>() << "class " << result[i][5].item<int>() << std::endl;
            std::cout << "ITERATION " << i << std::endl;

            REQUIRE(x == coordinate[indexObjectThres][0]);
            REQUIRE(y == coordinate[indexObjectThres][1]);
            REQUIRE(width == coordinate[indexObjectThres][2]);
            REQUIRE(height == coordinate[indexObjectThres][3]);
            REQUIRE(cls == coordinate[indexObjectThres][4]);
            indexObjectThres = indexObjectThres + 1;
        }
    }

    yolov4.reset();
    cudaDeviceReset();
}

#endif