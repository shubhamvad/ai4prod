#include <iostream>
#include "torch/torch.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "define.h"
#include "utils.h"

#pragma once
using namespace cv;

namespace aiProductionReady
{

//include guards
//#ifndef AIPRODMODELINTERFACE
//#define AIPRODMODELINTERFACE
    //classe interfaccia su come devono essere create le varie classi
    class modelInterface
    {

    public:
        virtual torch::Tensor preprocessing(Mat &Image) = 0;

        virtual torch::Tensor runmodel(torch::Tensor &input) = 0;

        virtual torch::Tensor postprocessing(torch::Tensor &output) = 0;
    };

//#endif

} // namespace aiProductionReady