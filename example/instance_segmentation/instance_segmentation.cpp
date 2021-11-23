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


#include <bits/stdc++.h>
#include <iostream>

#include "torch/torch.h"

#include "instancesegmentation.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// include namespace 
using namespace ai4prod;

using namespace instanceSegmentation;
using namespace cv;
using namespace std::chrono;
using namespace torch::indexing;

using namespace std::chrono;

//this is needed if you want to scan and entire folder
// img1.jpg
// img2.jpg
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;

int main()
{	

	//Time Count
	clock_t startClock, endClock;


    Yolact yolact;
    
   	

	//You need to call init for every new class.
	//This initizalize class component.
	//This function return True if everything is initialized correctly

	//parameter Description: 
	//init(path_to_onnx_model,image_resize_w,image_resize_h,number_of_model_classes, numeber_of_Output_from_model,Mode,path_to_tensorrt_model)
    
	//Mode: Cpu, TensorRT
	//path_to_tensorrt_model: Path where the tensorrt optimized engine is saved
	
	//CHANGE THIS VALUE WITH YOURS
	yolact.init("../yolact.onnx",550,550,80,TensorRT,"../yolactTensorRT");	


	Mat img;
	img= imread("../yolact-federer.jpeg");	
	yolact.preprocessing(img);
	yolact.runmodel();

	/*
	result is a struct {
		
		torch::Tensor boxes; -> bbox need to be processed see getCorrectBbox()
        torch::Tensor masks; -> Mask need to be processed see getCorrectMask()
        torch::Tensor classes; -> Number of classes
        torch::Tensor scores; 
        torch::Tensor proto;

	}
	*/
	auto result = yolact.postprocessing();
	


	//return vector<Rect>
	auto resultBbox=yolact.getCorrectBbox(result);
	

	for(auto &rect: resultBbox){

		rectangle(img,rect,(255, 255, 255), 0.5);
	}

	imshow("bbox",img);
	//return vector<Mat> 

	auto start = high_resolution_clock::now();
	startClock = clock();
	auto resultMask = yolact.getCorrectMask(result);
	auto stop = high_resolution_clock::now();
	endClock = clock();

	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Mask time " << (float)duration.count() / 1000000 << "Sec" << endl;	

	double time_taken = double(endClock - startClock) / double(CLOCKS_PER_SEC);

	cout << "Mask Time Chrono " << fixed << time_taken << setprecision(5);
	cout << " sec " << endl;
	
	// add mask to same image
	cv::Mat fullMask= resultMask[0] | resultMask[1] | resultMask[2];
	for(auto& mask:resultMask){
		
		
		// cv::imshow("mask",mask);
		// waitKey(0);
	}

		cv::imwrite("../maskFull.png",fullMask);
		waitKey(0);
	
	
	
}
