#include "modelInterface.h"

using namespace std::chrono;

namespace ai4prod
{

    namespace poseEstimation
    {

        class AIPRODUCTION_EXPORT Hrnet : ai4prod::modelInterfacePoseEstimation
        {
        private:
            ai4prod::aiutils m_aut;
            MODE m_eMode;
            std::string m_sMessage;

            //Config Parameter
            YAML::Node m_ymlConfig;
            std::string m_sModelTrPath;
            std::string m_sModelOnnxPath;
            std::string m_sEngineFp;
            std::string m_sEngineCache;

            //neural network input dimension

            int m_iInput_h;
            int m_iInput_w;
            //original image width and height
            int m_iMrows;
            int m_iMcols;

            int m_iNumClasses;
            float m_fNmsThresh;
            float m_fDetectionThresh;

            //ONNXRUNTIME
            //onnxRuntime Session

            Ort::SessionOptions m_OrtSessionOptions;
            Ort::AllocatorWithDefaultOptions allocator;

            std::unique_ptr<Ort::Session> m_OrtSession;
            std::unique_ptr<Ort::Env> m_OrtEnv;

            //OnnxRuntime Input Model

            size_t m_num_input_nodes;
            std::vector<const char *> m_input_node_names;

            //OnnxRuntime Output Model

            size_t m_num_out_nodes;
            std::vector<const char *> m_output_node_names;

            //------------------METHOD------------------------------
            void setOnnxRuntimeEnv();
            bool checkParameterConfig(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path);
            bool createYamlConfig(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path);
            //void setEnvVariable();
            void setSession();
            void setOnnxRuntimeModelInputOutput();

        public:
            Hrnet();
            ~Hrnet();
            bool init(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path = NULL);
            void preprocessing(cv::Mat &Image);
            void runmodel();
            torch::Tensor postprocessing(std::string imagePathAccuracy = "");

        }; //Hrnet
    }      //Pose Estimation
} //Ai4prod
