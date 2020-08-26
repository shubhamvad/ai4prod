#include "modelInterface.h"
#include "utils.h"

namespace classification{


class ResNet50 : modelInterface{



    private:

    Ort::SessionOptions session_options;
    //la sessione deve essere inizializzata nel costruttore 
    Ort::Session *session;
    
    

    public:
    
    ResNet50();
    ResNet50(std::string modelPath);
    //il distruttore virtual permette di avere una migliore gestione della memoria evitando memory leak
    virtual ~ResNet50();
    
    torch::Tensor preprocessing(Mat &Image);
    torch::Tensor postprocessing(torch::Tensor &input);
    torch::Tensor runmodel(torch::Tensor &output);

   
};





}