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
using namespace cv;

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;


int main(){

    //YOLO MAP ---------------------------------------------------------------------
    
    //setup image folder of Coco dataset

    std::string AccurayFolderPath = "../../../Images/objectdetection/";

    Yolov3 *yolov3;

    

    // Check our api for full description 
    
    yolov3 = new Yolov3();
	cout << "INIT SESSION " << endl;

	// init(path_to_onnx_yolov3model.onnx,imageWidth,imageHeight,Mode,PathsavedModelandConfig)
	//Mode= Cpu,TensorRT

    //init function return true if model is initialized correctly
	// For model path is better to use absolute path in order to avoid loading different model
    yolov3->init("../../../../Model/Yolov3/yolov3-spp.onnx", 608, 608, TensorRT, "../tensorrtModel");

    cout<<"START PROCESSING"<<endl;

    //auto start = high_resolution_clock::now();

    double numDetection=0;

    for (const auto &entry : fs::directory_iterator(AccurayFolderPath))
    {
        //std::cout << entry.path() << std::endl;

        string image_id = entry.path().string();

        //read image
        Mat img;
        img = imread(image_id.c_str());

        // auto start1 = high_resolution_clock::now();

        yolov3->preprocessing(img);
       
        yolov3->runmodel();

		//result is multi dim tensor with n row.
		//n are number of detections
		//reuslt is (x,y,width,height,class_number,class_probability)
        torch::Tensor result = yolov3->postprocessing();

		
        // auto stop1 = high_resolution_clock::now();

        // auto duration1 = duration_cast<microseconds>(stop1 - start1);

        // cout << "SINGLE TIME INFERENCE " << (double)duration1.count() / (1000000) << "Sec" << endl;

     

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
                
                string category= to_string(result[i][4].item<int>());
                cv::rectangle(img, brect, cv::Scalar(255, 0, 0));
                cv::putText(img,                         //target image
                            category.c_str(),            //text
                            cv::Point(brect.x, brect.y), //top-left position
                            cv::FONT_HERSHEY_DUPLEX,
                            1.0,
                            CV_RGB(0, 0, 0), //font color
                            2);
                //put text on rect https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
            }

			cvtColor(img, img, cv::COLOR_RGB2BGR);
            imshow("immagine", img);
            waitKey(0);
        }
    }

    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);

    // cout << "SINGLE TIME INFERENCE " << (double)duration.count() / (1000000 * numDetection) << "Sec" << endl;



    cout<<"program end"<<endl;
    return 0;


}