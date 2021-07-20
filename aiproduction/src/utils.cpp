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

#include "utils.h"

//using namespace cv;

//LIBTORCH

namespace ai4prod
{

    /*
Convert a cv::Mat to torch::Tensor 
image: cvMat in BGR format
width: image_width
height: image_height
channel: channel image
batch: batch size for inference

Each pixel value is in range [0,1]
return: torch::Tensor with the right order of input dimension(B,C,W,H)

*/
    torch::Tensor aiutils::convertMatToTensor(cv::Mat ImageBGR, int width, int height, int channel, int batch, bool gpu)
    {

        cv::cvtColor(ImageBGR, ImageBGR, cv::COLOR_BGR2RGB);
        cv::Mat img_float;

        //Conversione dei valori nell'intervallo [0,1] tipico del tensore di Pytorch
        ImageBGR.convertTo(img_float, CV_32F, 1.0 / 255);

        auto img_tensor = torch::from_blob(img_float.data, {batch, height, width, channel}).to(torch::kCPU);

        // you need to be contiguous to have all address memory of tensor sequentially
        img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous();

        //std::cout << img_tensor.dim() << std::endl;
        //std::cout << img_tensor.sizes() << std::endl;

        return img_tensor;
    }

    /*
Convert a cv::Mat to torch::Tensor 
image: cvMat in BGR format
width: image_width
height: image_height
channel: channel image
batch: batch size for inference

Each pixel value is in range [0,255]
return: torch::Tensor with the right order of input dimension(B,C,W,H)

*/
    torch::Tensor aiutils::convertMatToTensor8bit(cv::Mat ImageBGR, int width, int height, int channel, int batch, bool gpu)
    {

        cv::cvtColor(ImageBGR, ImageBGR, cv::COLOR_BGR2RGB);
        cv::Mat img_float;
        
        ImageBGR.convertTo(img_float, CV_32F, 1.0);

        auto img_tensor = torch::from_blob(img_float.data, {batch, height, width, channel}).to(torch::kCPU);


        // you need to be contiguous to have all address memory of tensor sequentially
        img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous();

        //std::cout << img_tensor.dim() << std::endl;
        //std::cout << img_tensor.sizes() << std::endl;

        return img_tensor;
    }

    /*

return: Image BGR

*/

    cv::Mat aiutils::convertTensortToMat(torch::Tensor tensor, int width, int height)
    {

        //devo controllare che la dimensione di batch sia uguale a 1

        //squeeze(): rimuove tutti i valori con dimensione 1
        tensor = tensor.squeeze().detach().permute({1, 2, 0});
        tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
        //Devo convertire il tensore in Cpu se voglio visualizzarlo con OpenCv
        tensor = tensor.to(torch::kCPU);
        cv::Mat resultImg(height, width, CV_8UC3);
        std::memcpy((void *)resultImg.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());

        return resultImg;
    }

    cv::Mat aiutils::convertTensortToMat8bit(torch::Tensor tensor, int width, int height)
    {

        //devo controllare che la dimensione di batch sia uguale a 1

        //squeeze(): rimuove tutti i valori con dimensione 1
        tensor = tensor.squeeze().detach().permute({1, 2, 0});
        tensor = tensor.clamp(0, 255).to(torch::kU8);
        //Devo convertire il tensore in Cpu se voglio visualizzarlo con OpenCv
        tensor = tensor.to(torch::kCPU);
        cv::Mat resultImg(height, width, CV_8UC3);
        std::memcpy((void *)resultImg.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());

        return resultImg;
    }

    /*

verify if 2 Mat are equal

*/
    bool aiutils::equalImage(const cv::Mat &a, const cv::Mat &b)
    {
        if ((a.rows != b.rows) || (a.cols != b.cols))
            return false;
        cv::Scalar s = cv::sum(a - b);

        std::cout << s << std::endl;

        return (s[0] == 0) && (s[1] == 0) && (s[2] == 0);
    }

    torch::Tensor aiutils::convert2dVectorToTensor(std::vector<std::vector<float>> &input)
    {

        if (input[0].size() == 0)
        {

            std::cout << "empty value" << std::endl;
            auto tensor = torch::empty({200, 200, 3});
            return tensor;
        }

        else
        {

            cv::Mat NewSamples(0, input[0].size(), cv::DataType<float>::type);

            for (unsigned int i = 0; i < input.size(); ++i)
            {
                // Make a temporary cv::Mat row and add to NewSamples _without_ data copy
                cv::Mat Sample(1, input[0].size(), cv::DataType<float>::type, input[i].data());

                NewSamples.push_back(Sample);
            }

            torch::Tensor Output = torch::from_blob(NewSamples.data, {(long int)input.size(), (long int)input[0].size()}).contiguous().clone();

            return Output;
        }
    }

    // bool aiutils::checkFileExists(std::string Filename)
    // {
    //     return access(Filename.c_str(), 0) == 0;
    // }

    //convert string to enum
    MODE aiutils::setMode(std::string Mode)
    {

        if (Mode == "TensorRT")
        {
            
            return TensorRT;
        }

        if (Mode == "Cpu")
        {

            return Cpu;
        }

        if (Mode == "DirectML") {
            return DirectML;
        }
	
        return Default;
    }
    //convert enum to string
    std::string aiutils::setYamlMode(MODE t)
    {

        std::cout << "setYamMode " << t << std::endl;
        if (t == TensorRT)
        {

            return "TensorRT";
        }

        if (t == Cpu)
        {

            return "Cpu";
        }

        if (t == DirectML)
        {

            return "DirectML";
        }
	
	return "Default";
    }

    //This fuction check if Mode is implementend before instantiate onnxruntime Session
    bool aiutils::checkMode(MODE m, std::string &Message)
    {

        bool value;
        switch (m)
        {

        case TensorRT:
            Message = "Mode selected TensorRT";

            value = true;
            break;
        case Cpu:
            Message = "Mode selected Cpu";
            value = true;
            break;
        case DirectML:
            Message = "Mode selected DirectMl";
            value = true;
            break;
        default:
            Message = "Mode NOT IMPLEMENTED cannot continue";
            value = false;
            break;
        }

        return value;
    }

    bool aiutils::createFolderIfNotExist(std::string folderPath)
    {

        if (cv::utils::fs::exists(folderPath))
        {

            return true;
        }
        else
        {

            if (cv::utils::fs::createDirectory(folderPath))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }

} // namespace ai4prod
