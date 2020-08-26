#include "modelInterface.h"
#include "utils.h"

namespace classification{


class ResNet50 : modelInterface{



    private:

    //ONNX RUNTIME
    Ort::SessionOptions session_options;
    //la sessione deve essere inizializzata nel costruttore 
    Ort::Session *session;
    //env inizializzato nel costruttore
    Ort::Env *env;

    Ort::AllocatorWithDefaultOptions allocator;


    //OnnxRuntimeModelloInput

    size_t num_input_nodes;
    std::vector<const char *> input_node_names;
    std::vector<int64_t> input_node_dims;
    
    //OnnxRuntimeModelloOutput

    size_t num_out_nodes; 
    std::vector<const char *> out_node_names;
    std::vector<int64_t> out_node_dims;


    
    //Dimensione del tensore di input modello .onnx
    size_t input_tensor_size;

    float *p;

    torch::Tensor inputTensor;

   
    

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