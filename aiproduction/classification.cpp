#include "classification.h"
using namespace std;

namespace classification{


//Da fare 

/*
1) Costruttore inserire la path di caricamento del modello
2) Sistemare il codice per caricare il modello Resnet
*/

ResNet50::ResNet50(){


        //inizializzazione sessione OnnxRuntime
        //const char * model_path= "/home/tondelli/Desktop/2020/aiproductionready/onnxruntime/model/cpu/squeezenet.onnx";

        //Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "test");
        //session = new Ort::Session(env, model_path, session_options); 
        

        
        //session_options.SetIntraOpNumThreads(6);
        //session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);    
}

ResNet50::ResNet50(std::string path){

    Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "test");
    session = new Ort::Session(env, path.c_str(), session_options); 
        

        
    session_options.SetIntraOpNumThreads(6);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
}


/*
Distruttore
*/
ResNet50::~ResNet50(){



}


torch::Tensor ResNet50::preprocessing(Mat &Image){

    //ResNet50::model=data;

    torch::Tensor inputTensor;
    
    inputTensor=convertMatToTensor(Image,Image.cols,Image.rows,Image.channels(),1);
    
    Mat testImage;

    testImage=convertTensortToMat(inputTensor,640,480);

    imshow("test image",testImage);
    imshow("original",Image);
    waitKey(0);


    equalImage(Image,testImage);



    cout <<"preprocessing"<<endl;

    //cout<<session->GetInputCount()<<endl;

    //cout<<session.GetInputCount()<<endl;


    //Preprocessig Image

    
    return inputTensor;

}






torch::Tensor ResNet50::runmodel(torch::Tensor &input){

    //Conversione del tensore 


    torch::Tensor test= torch::rand({2, 3});
    return test;

}


torch::Tensor ResNet50::postprocessing(torch::Tensor &output){

    


    
    torch::Tensor test= torch::rand({2, 3});
    return test;


}




}