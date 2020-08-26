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
    
    resnet = new ResNet50();
    
    
    
    torch::Tensor test;
    
    
    Mat img;

    cout<<"immagine"<<endl;
    img=imread("/home/eric/Immagini/index.jpeg");

    imshow("test",img);

    waitKey(0);

    resize(img,img,Size(640,480),0.5,0.5,cv::INTER_LANCZOS4);

    test = resnet->preprocessing(img);
    
    //std::cout << test << std::endl;
    
    // cout<<resnet->model<<endl;
    // cout<< "test"<< endl;

    // customResnet cRes;

    // cRes.preprocessing(1);

    //torch::Tensor tensor = torch::rand({2, 3});
    //std::cout << tensor << std::endl;

    return 0;
}