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
#include "objectdetection.h"
#include "instancesegmentation.h"
#include "Sort.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// include namespace 
using namespace ai4prod;

using namespace instanceSegmentation;
using namespace cv;
using namespace std::chrono;
using namespace torch::indexing;

//this is needed if you want to scan and entire folder
// img1.jpg
// img2.jpg
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;

int main()
{	
	int test= 5 + 3*4;
	cout <<test<<endl;
	// auto test= torch::zeros({4,8});

	// auto prova= test >0.5;

	// cout<<prova<<endl;

	// auto test2= torch::rand({4,8,4});

	// test2=test2.index({prova});
	
	// cout<<"TEST 2 "<< test2<<endl;

	// cout<< "TEST "<< test<<endl;


	// test= test.index({
	// 	torch::indexing::Slice(None),
	// 	torch::indexing::Slice(None,2)
	// }).contiguous();

	// cout<<test.sizes()<<endl;

	// auto tMax= torch::max(test,0);
	
	// cout <<std::get<0>(tMax)<<endl;
	// //initialize resnet
    Yolact *yolact;
    
   	//create new instance
    yolact = new Yolact();

	//You need to call init for every new class.
	//This initizalize class component.
	//This function return True if everything is initialized correctly

	//parameter Description: 
	//init(path_to_onnx_model,image_resize_w,image_resize_h,number_of_model_classes, numeber_of_Output_from_model,Mode,path_to_tensorrt_model)
    
	//Mode: Cpu, TensorRT
	//path_to_tensorrt_model: Path where the tensorrt optimized engine is saved
	
	//CHANGE THIS VALUE WITH YOURS
	yolact->init("/home/aistudios/Develop/Official/Inprogress/Segmentation/yolact_onnx/yolact/yolact.onnx", 550, 550, Cpu, "../tensorrtModel"); 

	Mat img;
	img= imread("/home/aistudios/Develop/Official/Inprogress/Segmentation/yolact_onnx/yolact/image/yolact-nadal.jpeg");	
	yolact->preprocessing(img);
	yolact->runmodel();
	auto result = yolact->postprocessing();
	
	yolact->displayBbox(result,img);
	
	// //resnet = new ResNet50();
	// cout << "test" << endl;
	// //PATH TO FOLDER 
    // std::string AccurayFolderPath = "../../../../Images/classification/";

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
    //     resnet->preprocessing(img);

	// 	//run model on img
    //     resnet->runmodel();

	// 	//return a tuple<Tensor,Tensor>: <ClassId,Probability>
	// 	//This output is without softmax
    //     std::tuple<torch::Tensor, torch::Tensor> prediction = resnet->postprocessing();

	// cout << "TOP CLASS " << std::get<0>(prediction)[0].item<float>();
		
	// //std::get<0>(prediction)[0].item<float>();
	// //if You need softmax you can use Libtorch softmax

        

    
    // }
    cout<<"model Created"<<endl;
	getchar();
}
