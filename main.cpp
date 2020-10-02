#include <iostream>

#include "torch/torch.h"

#include "aiproduction/classification.h"
#include "aiproduction/objectdetection.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace aiProductionReady;
using namespace objectDetection;
using namespace classification;

using namespace std::chrono;

// #pragma comment(lib, "onnxruntime.lib")
// #pragma comment(lib, "user32.lib")
// #pragma comment(lib, "gdi32.lib")
//#pragma comment(lib, "onnxruntime_providers.lib")
//#pragma comment(lib, "onnxruntime_session.lib")
//#pragma comment(lib, "onnxruntime_optimizer.lib")
//#pragma comment(lib, "onnxruntime_mocked_allocator.lib")
//#pragma comment(lib, "onnxruntime_mlas.lib")
//#pragma comment(lib, "onnxruntime_graph.lib")
//#pragma comment(lib, "onnxruntime_common.lib")


//classe Custom posso cambiare solo il preprocessing lasciando invariato il resto
//Link lista funzioni che possono essere usate
// class customResnet : ResNet50{

//     public:

//     customResnet(){

//         cout<<"costruttoreCustom"<<endl;
//     }

//     torch::Tensor preprocessing(Mat &Image){

//        torch::Tensor test= torch::rand({2, 3});
//         return test;

//     }
// };

int main()
{

    cout<<"programma inizializato"<<endl;
   
    
	Yolov3 *yolov3;

	
	//linux
    //yolov3= new Yolov3("/home/aistudios/Develop/aiproductionready/onnxruntime/model/cpu/yolov3-spp-darknet.onnx",608,608,"/home/aistudios/1");

	//windows

	//C:\Users\erict\OneDrive\Desktop\Develop\aiproductionready\onnxruntime\models

	yolov3 = new Yolov3("C:/Users/erict/OneDrive/Desktop/Develop/aiproductionready/onnxruntime/models/yolov3-spp-darknet.onnx", 608, 608, "C:/Users/erict/OneDrive/Desktop/engine");
    
	//yolov3 = new Yolov3();
	Mat img;
    //linux
	//img=imread("/home/aistudios/Develop/aiproductionready/test/objectDetection/dog.jpg");

	//windows


	img = imread("C:/Users/erict/OneDrive/Desktop/Develop/aiproductionready/test/objectDetection/dog.jpg");
    //torch::Tensor test;


    auto start = high_resolution_clock::now();
    
    for(int i=0;i<100;i++){
    yolov3->preprocessing(img);
    yolov3->runmodel();
    
    
    torch::Tensor result = yolov3->postprocessing();
    
    }

     auto stop = high_resolution_clock::now(); 
    //cout << "Class " << std::get<0>(prediction)[0] << endl;

    auto duration = duration_cast<microseconds>(stop - start); 
    
    cout << "SINGLE TIME INFERENCE "<< (double)duration.count()/(1000000*100) << "Sec"<<endl;


    //    for (int i=0; i<result.sizes()[0];i++)
    //    {

    //        cv::Rect brect;
    //        cout << result << endl;

    //        float tmp[4] = {result[i][0].item<float>(), result[i][1].item<float>(), result[i][2].item<float>(), result[i][3].item<float>()};

           
    //        brect = yolov3->get_rect(img, tmp);

    //        cv::rectangle(img, brect, cv::Scalar(255, 0, 0));
           
    //        put text on rect https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
    //    }

       imshow("immagine", img);
       waitKey(0);
    // yolov3->runmodel(test);

    // ResNet50 *resnet;

    // resnet = new ResNet50("/home/aistudios/Develop/aiproductionready/onnxruntime/model/cpu/resnet.onnx", 1000, 5, "/home/aistudios/resnet");
    // Mat img;
    // img = imread("/home/aistudios/Develop/aiproductionready/test/classification/dog.jpeg");

    // auto start = high_resolution_clock::now(); 

    

    // for (int i=0; i<100;i++){


    // resnet->preprocessing(img);
    
    // resnet->runmodel();

    // std::tuple<torch::Tensor, torch::Tensor> prediction = resnet->postprocessing();
    // }

    // auto stop = high_resolution_clock::now(); 
    // //cout << "Class " << std::get<0>(prediction)[0] << endl;

    // auto duration = duration_cast<microseconds>(stop - start); 
    
    // cout << "SINGLE TIME INFERENCE "<< (double)duration.count()/(1000000*100) << "Sec"<<endl;

    
    
    
    
    // ResNet50 *resnet2;

    // resnet2 = new ResNet50("/home/aistudios/Develop/aiproductionready/onnxruntime/model/cpu/resnet.onnx","/home/aistudios/2");

    // torch::Tensor test;
    // torch::Tensor test2;
    // torch::Tensor out;

    // torch::Tensor out2;

    // Mat img;

    // cout<<"immagine"<<endl;
    // img=imread("/home/aistudios/Develop/aiproductionready/dog.jpeg");

    // imshow("test",img);

    // waitKey(0);

    // clock_t start,end;

    // resize(img,img,Size(224,224),0.5,0.5,cv::INTER_LANCZOS4);

    // test = resnet->preprocessing(img);

    // test2= resnet2->preprocessing(img);

    // try{
    // start = clock();
    // out= resnet->runmodel(test);
    // end = clock();
    // }
    // catch(...){

    //     cout<<"exception"<<endl;

    // }

    //  double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    //     cout << "TOTOAL TIME : " << fixed
    //          << time_taken << setprecision(5);
    //     cout << " sec " << endl;

    // clock_t start2,end2;

    // try{
    // start2 = clock();
    // out= resnet->runmodel(test);
    // end2 = clock();
    // }
    // catch(...){

    //     cout<<"exception"<<endl;

    // }

    //  double time_taken2 = double(end2 - start2) / double(CLOCKS_PER_SEC);
    //     cout << "TOTOAL TIME : " << fixed
    //          << time_taken2 << setprecision(5);
    //     cout << " sec " << endl;

    // cout<< "SECOND MODEL"<<endl;

    // out2= resnet2->runmodel(test2);

    //std::cout << test << std::endl;

    // cout<<resnet->model<<endl;
    // cout<< "test"<< endl;

    // customResnet cRes;

    // cRes.preprocessing(1);

    //torch::Tensor tensor = torch::rand({2, 3});
    //std::cout << tensor << std::endl;

    return 0;
}
