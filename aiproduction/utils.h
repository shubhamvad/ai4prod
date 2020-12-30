
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

#ifdef _WIN32
#include <io.h>
#define access _access_s
#else
#include <unistd.h>
#endif

using namespace cv;
using namespace std;

namespace aiProductionReady
{

   enum MODE
   {

      TensorRT,
      Cpu

   };

   //#ifndef AIPRODUTILS
   //#define AIPRODUTILS
   class aiutils

   {

   public:
      torch::Tensor convertMatToTensor(Mat ImageBGR, int width, int height, int channel, int batch, bool gpu = false);
      cv::Mat convertTensortToMat(torch::Tensor tensor, int width, int height);
      bool equalImage(const Mat &a, const Mat &b);
      torch::Tensor convert2dVectorToTensor(std::vector<std::vector<float>> &input);

      //handle Mode for YamlCpp
      MODE setMode(string Mode);
      string setYamlMode(MODE t);
      //utils file handling

      // bool checkFileExists(std::string Filename);

      inline bool checkFileExists(const std::string &name)
      {
         return (access(name.c_str(), F_OK) != -1);
      }

      bool checkMode(MODE m, string &Message);

      //#endif
   };

} // namespace aiProductionReady