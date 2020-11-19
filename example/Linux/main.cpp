#include <iostream>

#include "torch/torch.h"

#include "classification.h"
#include "objectdetection.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace aiProductionReady;
using namespace objectDetection;
using namespace classification;

using namespace std::chrono;

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;

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

cv::Mat padding(cv::Mat &img, int width, int height)
{

    int w, h, x, y;
    float r_w = width / (img.cols * 1.0);
    float r_h = height / (img.rows * 1.0);
    if (r_h > r_w)
    {
        w = width;
        h = r_w * img.rows;
        x = 0;
        y = (height - h) / 2;
    }
    else
    {
        w = r_h * img.cols;
        h = height;
        x = (width - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(height, width, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

int main()
{



    string testImage = "";

    std::string AccurayFolderPath = "/home/aistudios/Develop/ai4prod/classes/Coco/Val/val2017";

    cout << "programma inizializato" << endl;

    Yolov3 *yolov3;

    auto start1 = high_resolution_clock::now();
    //linux
    yolov3 = new Yolov3("/home/aistudios/Develop/aiproductionready/onnxruntime/model/cpu/yolov3-spp-darknet.onnx", 608, 608, TensorRT, "/home/aistudios/1");

    auto stop1 = high_resolution_clock::now();

    auto duration1 = duration_cast<microseconds>(stop1 - start1);

    cout << "tempo costruttore " << duration1.count() / (1000000) << endl;
    //windows

    //C:\Users\erict\OneDrive\Desktop\Develop\aiproductionready\onnxruntime\models

    //yolov3 = new Yolov3("C:/Users/erict/OneDrive/Desktop/Develop/aiproductionready/onnxruntime/models/yolov3-spp-darknet.onnx", 608, 608, "C:/Users/erict/OneDrive/Desktop/engine");

    for (const auto &entry : fs::directory_iterator(AccurayFolderPath))
    {
        std::cout << entry.path() << std::endl;

        string image_id = entry.path();

        cout << image_id << endl;

        Mat img;
        //linux
        //img = imread("/home/aistudios/Develop/aiproductionready/test/objectDetection/dog.jpg");

        //CRASH
        //img = imread("/home/aistudios/Develop/ai4prod/classes/Coco/Val/val2017/000000411754.jpg");

        //

        //img = imread("/home/aistudios/Develop/ai4prod/classes/Coco/Val/val2017/000000460494.jpg");

        //CICLO
        img = imread(image_id.c_str());

        //resize(img,img,Size(608,608),0.5,0.5,cv::INTER_LANCZOS4);

        //img=padding(img,608,608);

        //imshow("test", img);

        //waitKey(500);

        //yolov3 = new Yolov3();

        //windows

        //img = imread("C:/Users/erict/OneDrive/Desktop/Develop/aiproductionready/test/objectDetection/dog.jpg");
        //torch::Tensor test;

        auto start = high_resolution_clock::now();

        //for(int i=0;i<100;i++){
        yolov3->preprocessing(img);
        yolov3->runmodel();

        torch::Tensor result = yolov3->postprocessing();

        cout << "immagine preprocessata correttamente" << endl;
        //getchar();

        if (!result.numel())
        {
            std::cout << "tensor is empty!" << std::endl;
            // do other checks you wish to do
        }
        else
        {

            for (int i = 0; i < result.sizes()[0]; i++)
            {

                cv::Rect brect;
                cout << result << endl;

                float tmp[4] = {result[i][0].item<float>(), result[i][1].item<float>(), result[i][2].item<float>(), result[i][3].item<float>()};

                brect = yolov3->get_rect(img, tmp);

                cv::rectangle(img, brect, cv::Scalar(255, 0, 0));

                //put text on rect https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
            }

            imshow("immagine", img);
            waitKey(500);
        }
    }

    // //}

    // auto stop = high_resolution_clock::now();
    // //cout << "Class " << std::get<0>(prediction)[0] << endl;

    // auto duration = duration_cast<microseconds>(stop - start);

    // cout << "SINGLE TIME INFERENCE " << (double)duration.count() / (1000000 * 100) << "Sec" << endl;

    // for (int i = 0; i < result.sizes()[0]; i++)
    // {

    //     cv::Rect brect;
    //     cout << result << endl;

    //     float tmp[4] = {result[i][0].item<float>(), result[i][1].item<float>(), result[i][2].item<float>(), result[i][3].item<float>()};

    //     brect = yolov3->get_rect(img, tmp);

    //     cv::rectangle(img, brect, cv::Scalar(255, 0, 0));

    //     //put text on rect https://stackoverflow.com/questions/56108183/python-opencv-cv2-drawing-rectangle-with-text
    // }

    // imshow("immagine", img);
    // waitKey(0);
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

    // cout << "pre-delete-1" << endl;
    // //delete yolov3;

    // cout << "post-delete" << endl;

    // auto start2 = high_resolution_clock::now();

    // Yolov3 *yolov3_2;
    // yolov3_2 = new Yolov3("/home/aistudios/Develop/aiproductionready/onnxruntime/model/cpu/yolov3-spp-darknet.onnx", 608, 608, TensorRT, "/home/aistudios/3");

    // auto stop2 = high_resolution_clock::now();

    // auto duration2 = duration_cast<microseconds>(stop2 - start2);

    // cout << "tempo costruttore 2" << duration2.count() / (1000000) << endl;

    // cout << "pre-delet2" << endl;

    // delete yolov3;

    // cout << "exit" << endl;
    return 0;
}
