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

      

        //setup image folder of Coco dataset
        //Linux 
        //std::string AccurayFolderPath = "/media/aistudios/44c62318-a7de-4fb6-a3e2-01aba49489c5/Dataset/Coco2017/validation/val2017";
        //Windows
        //std::string AccurayFolderPath = "C:/Users/erict/Desktop/Dataset/Coco2017/validation/val2017";
        //Test Single Image
        std::string AccurayFolderPath = "/media/aistudios/44c62318-a7de-4fb6-a3e2-01aba49489c5/Develop/Official/ai4prod/testAccuracy/instanceSegmentation/TestCoco/Cpp/test";
        Yolact *yolact;

  
        // Check our api for full description
        // Yolact(path_to_onnx_yolov3model.onnx,imageWidth,imageHeight,numClasses,Mode,TensortFoldersavedModel)
        //MODE = (Cpu,TensorRt)
        yolact = new Yolact();

        bool checkInit=yolact->init("../yolact.onnx", 550, 550, 80, TensorRT, "../tensorrtModel2");
        
        if(!checkInit){
                
                cout << "Problem Init" << endl;

                return 0;
        }

        cout << "START PROCESSING" << endl;

        //auto start = high_resolution_clock::now();

        double numDetection = 0;

        vector<double> infTime;

        for (const auto &entry : fs::directory_iterator(AccurayFolderPath))
        {       
                std::cout << entry.path() << std::endl;

                string image_id = entry.path().string();

                //image processed
                //cout << image_id << endl;

                Mat img;

                img = imread(image_id.c_str());

                //duration0
                auto start0 = high_resolution_clock::now();
                
                yolact->preprocessing(img);
#ifdef TIME_EVAL
                //duration
                auto start = high_resolution_clock::now();

#endif
                yolact->runmodel();
#ifdef TIME_EVAL
                auto stop = high_resolution_clock::now();

                auto duration = duration_cast<microseconds>(stop - start);

#endif          
                //duration1
                auto start1 = high_resolution_clock::now();
                auto result = yolact->postprocessing(image_id);

#ifdef TIME_EVAL
                auto stop1 = high_resolution_clock::now();

                auto duration1 = duration_cast<microseconds>(stop1 - start);
                //total inference time
                auto duration0 = duration_cast<microseconds>(stop1 - start0);
                infTime.push_back((double)duration.count());
#endif

                // cout << "SINGLE TIME INFERENCE " << (double)duration.count() / (1000000) << "Sec" << endl;
                
#ifdef TIME_EVAL
                if (numDetection == 10)
                        break;
#endif
                numDetection++;

                //cout << numDetection << endl;
                if (!result.scores.numel())
                {
                        std::cout << "tensor is empty! No detection Found" << std::endl;
                }
                else
                {

                        //SHOW RESULT

                       auto resultBbox = yolact->getCorrectBbox(result);

                        for (auto &rect : resultBbox)
                        {

                                rectangle(img, rect, (255, 255, 255), 0.5);
                        }

                        imshow("bbox", img);

                        auto resultMask = yolact->getCorrectMask(result);

                        for (auto &mask : resultMask)

                        {
                                cv::imshow("mask", mask);
                                waitKey(0);
                        }
                }
        }

       
#ifdef TIME_EVAL

        double sum_of_elems = 0;
        sum_of_elems = std::accumulate(infTime.begin(), infTime.end(), 0);

        cout << "SINGLE TIME INFERENCE 1 " << sum_of_elems / (1000000 * 10) << "Sec" << endl;

#endif
        //create yoloVal.json
        yolact->createAccuracyFile();

        cout << "program end" << endl;
        return 0;
}
