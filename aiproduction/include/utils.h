
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

#pragma once

#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>


#ifdef _WIN32
#include <io.h>
#define access _access_s
#else
#include <unistd.h>
#endif

//using namespace cv;


namespace ai4prod
{

   enum MODE
   {

      TensorRT,
      Cpu,
      DirectML

   };

   //#ifndef AIPRODUTILS
   //#define AIPRODUTILS
   class aiutils

   {

   public:
      torch::Tensor convertMatToTensor(cv::Mat ImageBGR, int width, int height, int channel, int batch, bool gpu = false,bool rgb=true);
      torch::Tensor convertMatToTensor8bit(cv::Mat ImageBGR, int width, int height, int channel, int batch, bool gpu = false );
      cv::Mat convertTensortToMat8bit(torch::Tensor tensor, int width, int height);
      cv::Mat convertTensortToMat(torch::Tensor tensor, int width, int height);
      bool equalImage(const cv::Mat &a, const cv::Mat &b);
      torch::Tensor convert2dVectorToTensor(std::vector<std::vector<float>> &input);

      //handle Mode for YamlCpp
      MODE setMode(std::string Mode);
      std::string setYamlMode(MODE t);

      bool createFolderIfNotExist(std::string folderPath);
      //utils file handling

      // bool checkFileExists(std::string Filename);
#ifdef __linux__
      inline bool checkFileExists(const std::string &name)
      {
         return (access(name.c_str(), F_OK) != -1);
      }
#elif _WIN32


	  inline bool checkFileExists(const std::string& name) {
          std::ifstream f(name.c_str());
		  return f.good();
	  }


#endif
      bool checkMode(MODE m, std::string &Message);

      //#endif
   };

} // namespace ai4prod