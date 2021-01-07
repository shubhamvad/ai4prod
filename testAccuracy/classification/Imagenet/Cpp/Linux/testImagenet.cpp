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

    ResNet50 *resnet;

    //See our example classification for full description of parameters
    int i = 0;
    resnet = new ResNet50();

    cout << "INIT SESSION: Could take some time if TensorRT Mode selected" << endl;
    resnet->init("../../../../../Model/Resnet50/resnet50.onnx", 256, 256, 1000, 5, TensorRT, "tensorrtModel");
    //resnet = new ResNet50();
    std::string AccurayFolderPath = "../../../../../Dataset/Imagenet2012/val2012";

    vector<double> infTime;

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
#ifdef TIME_EVAL

        auto start = high_resolution_clock::now();

#endif

        resnet->runmodel();

#ifdef TIME_EVAL
        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(stop - start);

        infTime.push_back((double)duration.count());
#endif

        std::tuple<torch::Tensor, torch::Tensor> prediction = resnet->postprocessing();



        // auto stop = high_resolution_clock::now();

        // auto duration = duration_cast<microseconds>(stop - start);

        // cout << "SINGLE TIME INFERENCE 1 " << (double)duration.count() / (1000000) << "Sec" << endl;
#ifdef TIME_EVAL
        if (i == 1000)
            break;
#endif

        i = i + 1;
    }

#ifdef TIME_EVAL

    double sum_of_elems = 0;
    sum_of_elems = std::accumulate(infTime.begin(), infTime.end(), 0);

    cout << "SINGLE TIME INFERENCE 1 " << sum_of_elems / (1000000 * 1000) << "Sec" << endl;

#endif
}