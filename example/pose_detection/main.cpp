#include <iostream>

#include "torch/torch.h"

#include "classification.h"
#include "objectdetection.h"
#include "posedetection.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace ai4prod;
using namespace objectDetection;
using namespace poseDetection;
using namespace classification;

using namespace std::chrono;

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;
using namespace cv;

int main(){


    Hrnet *hrnet;

    hrnet= new Hrnet();

    hrnet->init("/media/aistudios/44c62318-a7de-4fb6-a3e2-01aba49489c5/Develop/Official/ai4prod/example/pose_detection/hrnet.onnx",192,256,80,TensorRT,"../tensorrtModel");
    

    cout<< "program end"<< endl;
    return 0;


}