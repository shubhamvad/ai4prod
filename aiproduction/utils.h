
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

      bool checkFileExists(std::string Filename);

      //#endif
   };

} // namespace aiProductionReady