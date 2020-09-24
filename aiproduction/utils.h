
#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#pragma once
using namespace cv;
using namespace std;


namespace aiProductionReady{

//#ifndef AIPRODUTILS
//#define AIPRODUTILS
class aiutils

{

public:
    torch::Tensor convertMatToTensor(Mat ImageBGR, int width, int height, int channel, int batch, bool gpu = false);
    cv::Mat convertTensortToMat(torch::Tensor tensor, int width, int height);
    bool equalImage(const Mat &a, const Mat &b);
    torch::Tensor convert2dVectorToTensor(std::vector<std::vector<float>>& input);



//#endif
};

}