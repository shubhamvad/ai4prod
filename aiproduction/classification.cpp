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
        const char * model_path= "onnxruntime/model/cpu/squeezenet.onnx";

        Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "test");
        session = new Ort::Session(env, model_path, session_options); 
        

        
        session_options.SetIntraOpNumThreads(6);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    
    
}

torch::Tensor ResNet50::postprocessing(torch::Tensor &output){

    


    
    torch::Tensor test= torch::rand({2, 3});
    return test;


}

torch::Tensor ResNet50::runmodel(torch::Tensor &input){

    //Conversione del tensore 


    torch::Tensor test= torch::rand({2, 3});
    return test;

}


torch::Tensor ResNet50::preprocessing(Mat &Image){

    //ResNet50::model=data;

    cout <<"preprocessing"<<endl;

    cout<<session->GetInputCount()<<endl;

    //cout<<session.GetInputCount()<<endl;

    torch::Tensor test= torch::rand({2, 3});
    return test;

}



}