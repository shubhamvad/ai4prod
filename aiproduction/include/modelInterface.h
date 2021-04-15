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


#ifdef _WIN32

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#endif // _WIN32


#pragma once


#include "yaml-cpp/yaml.h"
#include <iostream>
#include "torch/torch.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include "utils.h"
//Cmake generated
#include "aiproduction_export.h"
#include "customDataType.h"



#include <json/json.h>

#include <experimental/filesystem>
#include <algorithm>

namespace fs = std::experimental::filesystem;
using namespace torch::indexing;

//using namespace cv;

namespace ai4prod
{

//include guards
//#ifndef AIPRODMODELINTERFACE
//#define AIPRODMODELINTERFACE
    //classe interfaccia su come devono essere create le varie classi
    class modelInterfaceClassification
    {

    public:
        virtual void preprocessing(cv::Mat &Image) = 0;

        virtual void runmodel() = 0;

        virtual std::tuple<torch::Tensor,torch::Tensor> postprocessing()=0; 
    };


     class modelInterfaceObjectDetection
    {

    public:
        virtual void preprocessing(cv::Mat &Image) = 0;

        virtual void runmodel() = 0;

        virtual torch::Tensor postprocessing(std::string imagePathAccuracy="")=0;
    };

    class modelInterfaceInstanceSegmentation{

        
        virtual void preprocessing(cv::Mat &Image) = 0;

        virtual void runmodel() = 0;

        virtual InstanceSegmentationResult postprocessing(std::string imagePathAccuracy="")=0;

    };

//#endif

} // namespace ai4prod
