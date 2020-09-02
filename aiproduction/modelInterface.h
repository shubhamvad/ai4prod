#include <iostream>
#include "torch/torch.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "define.h"



using namespace cv;

//classe interfaccia su come devono essere create le varie classi
class modelInterface{

public:

virtual torch::Tensor preprocessing(Mat &Image)=0;

virtual torch::Tensor runmodel(torch::Tensor &input)=0;

virtual torch::Tensor postprocessing(torch::Tensor &output)=0;

};