#include <iostream>
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

int main()
{

    ResNet50 *resnet;

    //See our example classification for full description of parameters
    resnet = new ResNet50("/home/aistudios/Develop/ai4prod/deps/onnxruntime/model/cpu/resnet50.onnx", 1000, 5, TensorRT, "/home/aistudios/resnetNew");

    std::string AccurayFolderPath = "/home/aistudios/Develop/ai4prod/classes/imagenet/Val/ILSVRC2012_img_val";

    for (const auto &entry : fs::directory_iterator(AccurayFolderPath))
    {

        string image_id = entry.path();
        Mat img;

        img = imread(image_id.c_str());

        //this is needed to make image_id matching with the one that is currently processed in the csv file
        resnet->m_sAccurayImagePath = image_id.c_str();

        resnet->preprocessing(img);

        resnet->runmodel();

        std::tuple<torch::Tensor, torch::Tensor> prediction = resnet->postprocessing();
    }

}