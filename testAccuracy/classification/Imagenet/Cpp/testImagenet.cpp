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
    int i = 0;
    resnet = new ResNet50();

    resnet->init("/home/aistudios/Develop/ai4prod/deps/onnxruntime/model/cpu/resnet50.onnx", 256, 256, 1000, 5, Cpu, "/home/aistudios/7");
    //resnet = new ResNet50();
    std::string AccurayFolderPath = "/home/aistudios/Develop/ai4prod/classes/imagenet/Val/ILSVRC2012_img_val";

    for (const auto &entry : fs::directory_iterator(AccurayFolderPath))
    {

        string image_id = entry.path();
        Mat img;
        cout << i << endl;
        img = imread(image_id.c_str());

        // auto start = high_resolution_clock::now();


        //this is needed to make image_id matching with the one that is currently processed in the csv file
        resnet->m_sAccurayImagePath = image_id.c_str();

        resnet->preprocessing(img);

        resnet->runmodel();

        std::tuple<torch::Tensor, torch::Tensor> prediction = resnet->postprocessing();

        // auto stop = high_resolution_clock::now();

        // auto duration = duration_cast<microseconds>(stop - start);

        // cout << "SINGLE TIME INFERENCE 1 " << (double)duration.count() / (1000000) << "Sec" << endl;

        i = i + 1;
    }
}