
#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#pragma once
using namespace cv;


namespace aiProductionReady{

//#ifndef AIPRODUTILS
//#define AIPRODUTILS
class aiutils

{

public:
    torch::Tensor convertMatToTensor(Mat ImageBGR, int width, int height, int channel, int batch, bool gpu = false);
    cv::Mat convertTensortToMat(torch::Tensor tensor, int width, int height);
    bool equalImage(const Mat &a, const Mat &b);
    inline std::vector<uint64_t> nms(const std::vector<std::array<float, 4>>& bboxes,            //
                                 const std::vector<float>& scores,                           //
                                 const float overlapThresh = 0.45,                           //
                                 const uint64_t topK = std::numeric_limits<uint64_t>::max()  //
);

template <typename T> std::deque<size_t> sortIndexes(const std::vector<T>& v);

};


//#endif
}