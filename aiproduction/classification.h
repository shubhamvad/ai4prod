#include "modelInterface.h"

namespace aiProductionReady
{

   namespace classification
   {

      class AIPRODUCTION_EXPORT ResNet50 : aiProductionReady::modelInterfaceClassification
      {

      private:
         //ONNX RUNTIME
         Ort::SessionOptions m_OrtSessionOptions;
         //la sessione deve essere inizializzata nel costruttore
         std::unique_ptr<Ort::Session> m_OrtSession;
         //env inizializzato nel costruttore
         std::unique_ptr<Ort::Env> m_OrtEnv;

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

         //

         int m_iModelNumberOfClass;
         int m_iNumberOfReturnedPrediction;

         //puntatori dati uscita ingresso onnxruntime
         float *m_fpOutOnnxRuntime;
         float *m_fpInOnnxRuntime;

         torch::Tensor inputTensor;
         torch::Tensor m_TOutputTensor;

         //path del modello di tensorrt
         std::string m_sModelTrPath;

         aiProductionReady::aiutils aut;
         //Config
        YAML::Node m_ymlConfig;

      public:
         ResNet50();
         ResNet50(std::string modelPath, int ModelNumberOfClass, int NumberOfReturnedPrediction, MODE t, std::string modelTr_path = NULL);
         //il distruttore virtual permette di avere una migliore gestione della memoria evitando memory leak
         virtual ~ResNet50();

         string m_sAccurayImagePath;

         void preprocessing(Mat &Image);
         std::tuple<torch::Tensor, torch::Tensor> postprocessing();
         void runmodel();
      };

   } //namespace classification

} // namespace aiProductionReady