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
    class modelInterfaceClassification
    {

    public:
        virtual void preprocessing(Mat &Image) = 0;

        virtual void runmodel() = 0;

        virtual std::tuple<torch::Tensor,torch::Tensor> postprocessing()=0;
    };


     class modelInterfaceObjectDetection
    {

    public:
        virtual void preprocessing(Mat &Image) = 0;

        virtual void runmodel() = 0;

        virtual torch::Tensor postprocessing()=0;
    };

//#endif

} // namespace aiProductionReady