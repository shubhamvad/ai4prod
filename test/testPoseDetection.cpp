
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

#include "classification.h"
#include "objectdetection.h"
#include "posedetection.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace ai4prod;
using namespace objectDetection;
using namespace poseDetection;

using namespace std::chrono;

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;
using namespace cv;

#if defined AICPU || defined TENSORRT

TEST_CASE("Init Pose Detection Hrnet Cpu")
{
    std::cout << "TEST INIT HRNET ON CPU" << std::endl;

    Hrnet hrnet;
    bool initFlag = false;
    initFlag = hrnet.init("../Model/hrnet.onnx", 256, 192, 80, Cpu,
                          "../test/testHrnetCpu");

    REQUIRE(initFlag == true);
}

TEST_CASE("Test Pose Detection Output Hrnet Cpu")
{
    std::cout << "TEST POSE DETECTION OUTPUT Hrnet CPU" << std::endl;

    Hrnet hrnet;
    Yolov4 yolov4;

    hrnet.init("../Model/hrnet.onnx", 256, 192, 80, Cpu,
               "../test/testHrnetCpu");

    std::string image_id = "../Images/posedetection/sinner.jpg";

    cv::Mat image = cv::imread(image_id.c_str());

    vector<int> includeClass = {0};
    yolov4.init("../Model/yolov4_608.onnx", 608, 608, 80, Cpu, "../test/testYolov4Cpu", &includeClass);

    yolov4.preprocessing(image);

    yolov4.runmodel();

    torch::Tensor result = yolov4.postprocessing();

    std::vector<cv::Point2i> testCoordinate = {cv::Point2i(850, 267), cv::Point2i(869, 248), cv::Point2i(850, 248),
                                               cv::Point2i(945, 267), cv::Point2i(850, 286), cv::Point2i(1041, 363), cv::Point2i(812, 382), cv::Point2i(1232, 382), cv::Point2i(601, 420),
                                               cv::Point2i(1404, 477), cv::Point2i(430, 382), cv::Point2i(1155, 592), cv::Point2i(1079, 630), cv::Point2i(907, 726), cv::Point2i(1346, 783),
                                               cv::Point2i(754, 993), cv::Point2i(1633, 802)};

    for (int i = 0; i < result.sizes()[0]; i++)
    {
        int x = result[i][0].item<int>();
        int y = result[i][1].item<int>();
        int width = result[i][2].item<int>();
        int height = result[i][3].item<int>();

        cv::Rect brect = cv::Rect(x, y, width, height);

        hrnet.preprocessing(image, brect);

        hrnet.runmodel();

        torch::Tensor poseResult = hrnet.postprocessing();

        for (int j = 0; j < poseResult.sizes()[0]; j++)
        {
            for (int i = 0; i < 17; i++)
            {

                cv::Point2i joint = cv::Point2i(poseResult[j][i][0].item<int>(), poseResult[j][i][1].item<int>());

                std::cout << " ITERATION " << i << std::endl;

                REQUIRE(joint.x == testCoordinate[i].x);
                REQUIRE(joint.y == testCoordinate[i].y);
            }
        }
    }
}

#endif

#if defined(TENSORRT)

TEST_CASE("Init Pose Detection Hrnet TensorRT")
{
    std::cout << "TEST INIT HRNET ON Tensorrt" << std::endl;

    std::unique_ptr<Hrnet> hrnet = std::make_unique<Hrnet>();
    bool initFlag = false;
    initFlag = hrnet->init("../Model/hrnet.onnx", 256, 192, 80, TensorRT,
                           "../test/testHrnetTensorRT");

    hrnet.reset();
    REQUIRE(initFlag == true);
}

TEST_CASE("Test Pose Detection Output Hrnet TensorRT")
{
    std::cout << "TEST POSE DETECTION OUTPUT Hrnet TensorRT" << std::endl;

    std::unique_ptr<Hrnet> hrnet = std::make_unique<Hrnet>();
    std::unique_ptr<Yolov4> yolov4 = std::make_unique<Yolov4>();

    hrnet->init("../Model/hrnet.onnx", 256, 192, 80, TensorRT,
                "../test/testHrnetTensorRT");

    std::string image_id = "../Images/posedetection/sinner.jpg";

    cv::Mat image = cv::imread(image_id.c_str());

    vector<int> includeClass = {0};
    yolov4->init("../Model/yolov4_608.onnx", 608, 608, 80, TensorRT, "../test/testYolov4TensorRt", &includeClass);

    yolov4->preprocessing(image);

    yolov4->runmodel();

    torch::Tensor result = yolov4->postprocessing();

    std::vector<cv::Point2i> testCoordinate = {cv::Point2i(850, 267), cv::Point2i(869, 248), cv::Point2i(850, 248),
                                               cv::Point2i(945, 267), cv::Point2i(850, 286), cv::Point2i(1041, 363), cv::Point2i(812, 382), cv::Point2i(1232, 382), cv::Point2i(601, 420),
                                               cv::Point2i(1404, 477), cv::Point2i(430, 382), cv::Point2i(1155, 592), cv::Point2i(1079, 630), cv::Point2i(907, 726), cv::Point2i(1346, 783),
                                               cv::Point2i(754, 993), cv::Point2i(1633, 802)};

    for (int i = 0; i < result.sizes()[0]; i++)
    {
        int x = result[i][0].item<int>();
        int y = result[i][1].item<int>();
        int width = result[i][2].item<int>();
        int height = result[i][3].item<int>();

        cv::Rect brect = cv::Rect(x, y, width, height);

        hrnet->preprocessing(image, brect);

        hrnet->runmodel();

        torch::Tensor poseResult = hrnet->postprocessing();

        for (int j = 0; j < poseResult.sizes()[0]; j++)
        {
            for (int i = 0; i < 17; i++)
            {

                cv::Point2i joint = cv::Point2i(poseResult[j][i][0].item<int>(), poseResult[j][i][1].item<int>());

                std::cout << " ITERATION " << i << std::endl;

                REQUIRE(joint.x == testCoordinate[i].x);
                REQUIRE(joint.y == testCoordinate[i].y);
            }
        }
    }

    yolov4.reset();
    hrnet.reset();
    cudaDeviceReset();
}

#endif