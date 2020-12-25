#include "modelInterface.h"

namespace aiProductionReady
{

   namespace classification
   {

      class AIPRODUCTION_EXPORT ResNet50 : aiProductionReady::modelInterfaceClassification
      {

      private:

         //INIT VARIABLE

         int m_iModelNumberOfClass;
         int m_iNumberOfReturnedPrediction;

         YAML::Node m_ymlConfig;
         std::string m_sModelTrPath;
         std::string m_sModelOnnxPath;
         std::string m_sEngineFp;
         std::string m_sEngineCache;

         MODE m_eMode;
         int m_iInput_h;
         int m_iInput_w;
         int m_iCropImage;


         //ONNX RUNTIME

         Ort::SessionOptions m_OrtSessionOptions;
         Ort::AllocatorWithDefaultOptions allocator;
         
         std::unique_ptr<Ort::Session> m_OrtSession;
         std::unique_ptr<Ort::Env> m_OrtEnv;

         
         //OnnxRuntimeModelloInput

         size_t num_input_nodes;
         std::vector<const char *> input_node_names;
         std::vector<int64_t> input_node_dims;

         //OnnxRuntimeModelloOutput

         size_t num_out_nodes;
         std::vector<const char *> out_node_names;
         std::vector<int64_t> out_node_dims;

         //onnxruntime data
         float *m_fpOutOnnxRuntime;
         float *m_fpInOnnxRuntime;

         //Dimensione del tensore di input modello .onnx
         size_t input_tensor_size;


         //LIBTORCH DATA

         torch::Tensor inputTensor;
         torch::Tensor m_TOutputTensor;


         //THREAD SAFE

         //handle initialization
         bool m_bInit;
         //used to call init only one time per instances
         bool m_bCheckInit;
         //used to verify if preprocess is called on the same run
         bool m_bCheckPre;
         //used to verify if run model is called on the same run
         bool m_bCheckRun;
         //used to verify id post process is called
         bool m_bCheckPost;

         aiProductionReady::aiutils aut;

      
         //ERROR HANDLING

         //error message
         string m_sMessage;


         // FUNCTION


         //init Function

         void setOnnxRuntimeEnv();
         void setOnnxRuntimeModelInputOutput();
         void createYamlConfig(std::string modelPath, int width, int height, int ModelNumberOfClass, int NumberOfReturnedPrediction, MODE t, std::string modelTr_path);
         void setEnvVariable();
         void setSession();



      public:
         ResNet50();

         virtual ~ResNet50();

         bool init(std::string modelPath, int width, int height, int ModelNumberOfClass, int NumberOfReturnedPrediction, MODE t, std::string modelTr_path = NULL);

         string m_sAccurayImagePath;

         void preprocessing(Mat &Image);
         std::tuple<torch::Tensor, torch::Tensor> postprocessing();
         void runmodel();
      };

   } //namespace classification

} // namespace aiProductionReady