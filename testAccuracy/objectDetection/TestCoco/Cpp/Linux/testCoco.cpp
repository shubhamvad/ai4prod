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

//inference time detection
#define TIME_EVAL

#include <iostream>

#include "torch/torch.h"

#include "classification.h"
#include "objectdetection.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace ai4prod;
using namespace objectDetection;
using namespace classification;

using namespace std::chrono;

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;

int main()
{

    //YOLO MAP ---------------------------------------------------------------------

    //setup image folder of Coco dataset

    std::string AccurayFolderPath = "/media/d50f8cb9-a942-4133-ba99-0a298b9dddd5/data/Coco2017/validation/val2017";

    Yolov3 *yolov3;

    //linux
    // Check our api for full description
    // Yolov3(path_to_onnx_yolov3model.onnx,imageWidth,imageHeight,Mode,TensortFoldersavedModel)
    yolov3 = new Yolov3();

    yolov3->init("../../../../../../Model/Yolov3/yolov3-spp.onnx", 608, 608, TensorRT, "tensorrtModel");
    //windows

    //C:\Users\erict\OneDrive\Desktop\Develop\ai4prod\onnxruntime\models

    //yolov3 = new Yolov3("C:/Users/erict/OneDrive/Desktop/Develop/ai4prod/onnxruntime/models/yolov3-spp-darknet.onnx", 608, 608, "C:/Users/erict/OneDrive/Desktop/engine");

    cout << "START PROCESSING" << endl;

    //auto start = high_resolution_clock::now();

    double numDetection = 0;

    vector<double> infTime;

    for (const auto &entry : fs::directory_iterator(AccurayFolderPath))
    {
        //std::cout << entry.path() << std::endl;

        string image_id = entry.path();

        //image processed
        //cout << image_id << endl;

        Mat img;

        img = imread(image_id.c_str());

        // auto start1 = high_resolution_clock::now();

        yolov3->m_sAccurayImagePath = image_id.c_str();

        yolov3->preprocessing(img);
#ifdef TIME_EVAL

        auto start = high_resolution_clock::now();

#endif
        yolov3->runmodel();
#ifdef TIME_EVAL
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);

        infTime.push_back((double)duration.count());
#endif

        torch::Tensor result = yolov3->postprocessing();

        // auto stop1 = high_resolution_clock::now();

        // auto duration1 = duration_cast<microseconds>(stop1 - start1);

        // cout << "SINGLE TIME INFERENCE " << (double)duration1.count() / (1000000) << "Sec" << endl;
#ifdef TIME_EVAL
        if (numDetection == 1000)
            break;
#endif
        numDetection++;

        // cout<<numDetection<<endl;
        if (!result.numel())
        {
            std::cout << "tensor is empty! No detection Found" << std::endl;
        }
        else
        {

            //if you want to see output results. This slow down the processing

            // for (int i = 0; i < result.sizes()[0]; i++)
            // {

            //     cv::Rect brect;
            //     //cout << result << endl;

            //     float tmp[4] = {result[i][0].item<float>(), result[i][1].item<float>(), result[i][2].item<float>(), result[i][3].item<float>()};

            //     brect = yolov3->get_rect(img, tmp);

            //     string category= to_string(result[i][4].item<float>());
            //     cv::rectangle(img, brect, cv::Scalar(255, 0, 0));
            //     cv::putText(img,                         //target image
            //                 category.c_str(),            //text
            //                 cv::Point(brect.x, brect.y), //top-left position
            //                 cv::FONT_HERSHEY_DUPLEX,
            //                 1.0,
            //                 CV_RGB(118, 185, 0), //font color
            //                 2);
            //     //put text on rect https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
            // }

            // imshow("immagine", img);
            // waitKey(0);
        }
    }

    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);

    // cout << "SINGLE TIME INFERENCE " << (double)duration.count() / (1000000 * numDetection) << "Sec" << endl;
#ifdef TIME_EVAL

    double sum_of_elems = 0;
    sum_of_elems = std::accumulate(infTime.begin(), infTime.end(), 0);

    cout << "SINGLE TIME INFERENCE 1 " << sum_of_elems / (1000000 * 1000) << "Sec" << endl;

#endif
    //create yoloVal.json
    yolov3->createAccuracyFile();

    cout << "program end" << endl;
    return 0;
}
