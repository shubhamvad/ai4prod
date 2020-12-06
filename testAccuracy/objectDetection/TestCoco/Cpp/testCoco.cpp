#include <iostream>

#include "torch/torch.h"

#include "classification.h"
#include "objectdetection.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace aiProductionReady;
using namespace objectDetection;
using namespace classification;

using namespace std::chrono;

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;


int main(){

    //YOLO MAP ---------------------------------------------------------------------
    
    //setup image folder of Coco dataset

    std::string AccurayFolderPath = "/home/aistudios/Develop/ai4prod/classes/Coco/Val/val2017";

    

    Yolov3 *yolov3;

    
    
    //linux
    // Check our api for full description 
    // Yolov3(path_to_onnx_yolov3model.onnx,imageWidth,imageHeight,Mode,TensortFoldersavedModel)
    yolov3 = new Yolov3("/home/aistudios/Develop/aiproductionready/onnxruntime/model/cpu/yolov3-spp-darknet.onnx", 608, 608, TensorRT, "/home/aistudios/1");

    //windows

    //C:\Users\erict\OneDrive\Desktop\Develop\aiproductionready\onnxruntime\models

    //yolov3 = new Yolov3("C:/Users/erict/OneDrive/Desktop/Develop/aiproductionready/onnxruntime/models/yolov3-spp-darknet.onnx", 608, 608, "C:/Users/erict/OneDrive/Desktop/engine");

    for (const auto &entry : fs::directory_iterator(AccurayFolderPath))
    {
        //std::cout << entry.path() << std::endl;

        string image_id = entry.path();

        //image processed
        //cout << image_id << endl;

        Mat img;
              
        img = imread(image_id.c_str());

        yolov3->m_sAccurayImagePath = image_id.c_str();

     
        yolov3->preprocessing(img);
        yolov3->runmodel();

        torch::Tensor result = yolov3->postprocessing();


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

    //create yoloVal.json
    yolov3->createAccuracyFile();

    cout<<"program end"<<endl;
    return 0;


}