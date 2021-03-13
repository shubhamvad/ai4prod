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

#include "instancesegmentation.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

using namespace ai4prod;
using namespace instanceSegmentation;

using namespace cv;

using namespace std::chrono;

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;

int main()
{

    //YOLO MAP ---------------------------------------------------------------------

    //setup image folder of Coco dataset

    std::string AccurayFolderPath = "/media/aistudios/44c62318-a7de-4fb6-a3e2-01aba49489c5/Dataset/Coco2017/validation/val2017";

    Yolact *yolact;

    //linux
    // Check our api for full description
    // Yolov3(path_to_onnx_yolov3model.onnx,imageWidth,imageHeight,Mode,TensortFoldersavedModel)
    yolact = new Yolact();

    yolact->init("/home/aistudios/Develop/Official/Inprogress/Segmentation/yolact_onnx/yolact/yolact.onnx", 550, 550, 80, TensorRT, "../tensorrtModel");
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

        yolact->preprocessing(img);
#ifdef TIME_EVAL

        auto start = high_resolution_clock::now();

#endif
        yolact->runmodel();
#ifdef TIME_EVAL
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);
 
        
        infTime.push_back((double)duration.count());
#endif
        auto start1 = high_resolution_clock::now();
        auto result = yolact->postprocessing(image_id);
        auto stop1 = high_resolution_clock::now();

        auto duration1 = duration_cast<microseconds>(stop1 - start1);

        // auto stop1 = high_resolution_clock::now();

        // auto duration1 = duration_cast<microseconds>(stop1 - start1);

        cout << "SINGLE TIME INFERENCE " << (double)duration.count() / (1000000) << "Sec" << endl;
        cout << "SINGLE TIME PostProcessing " << (double)duration1.count() / (1000000) << "Sec" << endl;
#ifdef TIME_EVAL
        if (numDetection == 1000)
            break;
#endif
        numDetection++;

        cout<<numDetection<<endl;
        if (result.boxes.sizes()[0] == 0)
        {
            std::cout << "tensor is empty! No detection Found" << std::endl;
        }
        else
        {
        
        //SHOW RESULT
        
        //     auto resultBbox = yolact->getCorrectBbox(result);

        //     for (auto &rect : resultBbox)
        //     {

        //         rectangle(img, rect, (255, 255, 255), 0.5);
        //     }

        //     imshow("bbox", img);

        //     auto resultMask = yolact->getCorrectMask(result);

        //     for (auto &mask : resultMask)
            
        //     {   cv::imshow("mask",mask);
        //         waitKey(0);

        //     }
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
        yolact->createAccuracyFile();

        cout << "program end" << endl;
        return 0;
    }
