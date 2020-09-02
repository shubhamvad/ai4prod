#include <iostream>

#include "torch/torch.h"

#include "aiproduction/classification.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace classification;
using namespace std;

//classe Custom posso cambiare solo il preprocessing lasciando invariato il resto
//Link lista funzioni che possono essere usate
class customResnet : ResNet50{

    public:

    customResnet(){

        cout<<"costruttoreCustom"<<endl;
    }

    torch::Tensor preprocessing(Mat &Image){

       torch::Tensor test= torch::rand({2, 3});
        return test;

    }
};

int main(){

    ResNet50 *resnet;

    resnet = new ResNet50("/home/aistudios/Develop/aiproductionready/onnxruntime/model/cpu/resnet.onnx");

    torch::Tensor test;
    torch::Tensor out;

    Mat img;

    cout<<"immagine"<<endl;
    img=imread("/home/aistudios/Develop/aiproductionready/dog.jpeg");

    imshow("test",img);

    waitKey(0);

    clock_t start,end;
    
    resize(img,img,Size(224,224),0.5,0.5,cv::INTER_LANCZOS4);

    test = resnet->preprocessing(img);

    try{
    start = clock();
    out= resnet->runmodel(test);
    end = clock();
    }
    catch(...){

        cout<<"exception"<<endl;

    }

     

     double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
        cout << "TOTOAL TIME : " << fixed
             << time_taken << setprecision(5);
        cout << " sec " << endl;


    clock_t start2,end2;


    try{
    start2 = clock();
    out= resnet->runmodel(test);
    end2 = clock();
    }
    catch(...){

        cout<<"exception"<<endl;

    }

     

     double time_taken2 = double(end2 - start2) / double(CLOCKS_PER_SEC);
        cout << "TOTOAL TIME : " << fixed
             << time_taken2 << setprecision(5);
        cout << " sec " << endl;    
   

    //std::cout << test << std::endl;

    // cout<<resnet->model<<endl;
    // cout<< "test"<< endl;

    // customResnet cRes;

    // cRes.preprocessing(1);

    //torch::Tensor tensor = torch::rand({2, 3});
    //std::cout << tensor << std::endl;

    return 0;
}

