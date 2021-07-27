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

#include <iostream>

#include "torch/torch.h"

#include "classification.h"
#include "explain.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// include namespace
using namespace ai4prod;
using namespace classification;
using namespace explain;
using namespace cv;
using namespace std::chrono;

//this is needed if you want to scan and entire folder
// img1.jpg
// img2.jpg
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;

// public used to access public method of ResNet50
class customResnet : public ResNet50
{
public:
	void preprocessing(cv::Mat &Image)
	{

		if (m_bInit && !m_bCheckPre && !m_bCheckRun && m_bCheckPost)
		{
			cout<<"NOT PARENT"<<endl;
			//resize(Image, Image, Size(256, 256), 0.5, 0.5, cv::INTER_LANCZOS4);
			inputTensor = aut.convertMatToTensor(Image, Image.cols, Image.rows, Image.channels(), 1);

			input_tensor_size = Image.cols * Image.rows * Image.channels();

			//define transform for CIFAR 10
			inputTensor[0][0] = inputTensor[0][0].sub_(0.4914).div_(0.2023);
			inputTensor[0][1] = inputTensor[0][1].sub_(0.4822).div_(0.1994);
			inputTensor[0][2] = inputTensor[0][2].sub_(0.4465).div_(0.2010);

			m_bCheckPre = true;
		}
		else
		{
			m_sMessage = "call init() before";
			std::cout << "call init() before" << std::endl;
		}
	}
};

int main()
{
	//initialize resnet
	customResnet custRes;
	//create new instance

	//You need to call init for every new class.
	//This initizalize class component.
	//This function return True if everything is initialized correctly

	//parameter Description:
	//init(path_to_onnx_model,image_resize_w,image_resize_h,number_of_model_classes, numeber_of_Output_from_model,Mode,path_to_tensorrt_model)

	//Mode: Cpu, TensorRT
	//path_to_tensorrt_model: Path where the tensorrt optimized engine is saved

	//CHANGE THIS VALUE WITH YOURS
	if (!custRes.init("../../models/Cirfar10-original.onnx", 32, 32, 10, 2, Cpu, "../cpuModel2"))
	{
		cout << "exit" << endl;
		getchar();
		return 0;
	}

	ConfusionMatrix cf;
	cout << "1" << endl;
	cf.init("../confMatrix23-07-2.csv", 10);

	// std::cout << cf.getConfutionMatrix() << std::endl;

	std::string AccurayFolderPath = "../images/";

	cout << "2" << endl;
	// read image from folder
	for (const auto &entry : fs::directory_iterator(AccurayFolderPath))
	{
		Mat img;
		string image_id = entry.path().string();
		//read image with opencv
		img = imread(image_id.c_str());
		custRes.preprocessing(img);
		
		custRes.runmodel();
		std::tuple<torch::Tensor, torch::Tensor> prediction = custRes.postprocessing();
		

		torch::Tensor CFProbability = cf.getProbability(std::get<0>(prediction));
		cout << "TOP CLASS " << std::get<0>(prediction)[0].item<float>()<<endl;
		
		cout << "PROBABILITY " << CFProbability[0].item<float>()<<endl;
		
		cout << "Neural network probability " << std::get<1>(prediction)[0].item<float>()<<endl;

		cout << "IMAGE NAME "<<image_id <<endl;


	}

	// //resnet = new ResNet50();
	// cout << "test" << endl;
	// //PATH TO FOLDER
	// std::string AccurayFolderPath = "../images/";

	// cout << "Start Classification" << endl;

	// for (const auto &entry : fs::directory_iterator(AccurayFolderPath))
	// {

	//     string image_id = entry.path().string();

	// 	//opencv Mat
	// 	Mat img;
	// 	//read image with opencv
	//     img = imread(image_id.c_str());

	// 	//ai4prod To understand how these functions works have look here https://www.ai4prod.ai/c-stack/

	// 	//preprocessing(cv::Mat)
	//     resnet.preprocessing(img);

	// 	//run model on img
	//     resnet.runmodel();

	// 	//return a tuple<Tensor,Tensor>: <ClassId,Probability>
	// 	//This output is without softmax
	//     std::tuple<torch::Tensor, torch::Tensor> prediction = resnet.postprocessing();

	// cout << "TOP CLASS " << std::get<0>(prediction)[0].item<float>();

	// //std::get<0>(prediction)[0].item<float>();
	// //if You need softmax you can use Libtorch softmax

	// }
	std::cout << "finish" << std::endl;
	getchar();
}
