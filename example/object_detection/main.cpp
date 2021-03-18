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
//#define TIME_EVAL

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

    //YOLO ---------------------------------------------------------------------

    //setup image folder

    std::string AccurayFolderPath = "/home/aistudios/Desktop/images_Manu/Immagini_cani";

    Yolov3 *yolov3;

    //linux
    // Check our api for full description
    // Yolov3(path_to_onnx_yolov3model.onnx,imageWidth,imageHeight,NumClasses,Mode,TensortFoldersavedModel)
    yolov3 = new Yolov3();

    if (!yolov3->init("/home/aistudios/Develop/Official/ai4prod/Model/Yolov3/yolov3-spp.onnx", 608, 608, 80, TensorRT, "tensorrtModel"))
    {

        return 0;
    }

    cout << "START PROCESSING" << endl;

    vector<double> infTime;

    for (const auto &entry : fs::directory_iterator(AccurayFolderPath))
    {

        string image_id = entry.path();

        Mat img;

        img = imread(image_id.c_str());

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

#ifdef TIME_EVAL
        if (numDetection == 1000)
            break;
#endif

        if (!result.numel())
        {
            std::cout << "tensor is empty! No detection Found" << std::endl;
        }
        else
        {

            //if you want to see output results. This slow down the processing

            for (int i = 0; i < result.sizes()[0]; i++)
            {

                cv::Rect brect;
                //cout << result << endl;

                float tmp[4] = {result[i][0].item<float>(), result[i][1].item<float>(), result[i][2].item<float>(), result[i][3].item<float>()};

                brect = yolov3->get_rect(img, tmp);

                string category = to_string(result[i][4].item<float>());
                cv::rectangle(img, brect, cv::Scalar(255, 0, 0));
                cv::putText(img,                         //target image
                            category.c_str(),            //text
                            cv::Point(brect.x, brect.y), //top-left position
                            cv::FONT_HERSHEY_DUPLEX,
                            1.0,
                            CV_RGB(118, 185, 0), //font color
                            2);
                //put text on rect https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
            }

            imshow("image", img);
            waitKey(0);
        }
    }

#ifdef TIME_EVAL

    double sum_of_elems = 0;
    sum_of_elems = std::accumulate(infTime.begin(), infTime.end(), 0);

    cout << "SINGLE TIME INFERENCE 1 " << sum_of_elems / (1000000 * 1000) << "Sec" << endl;

#endif

    cout << "program end" << endl;
    return 0;
}
