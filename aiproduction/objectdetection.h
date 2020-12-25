#include "modelInterface.h"

namespace aiProductionReady
{

    namespace objectDetection
    {

        class AIPRODUCTION_EXPORT Yolov3 : aiProductionReady::modelInterfaceObjectDetection
        {

        private:
            //INIT VARIABLE

            YAML::Node m_ymlConfig;
            std::string m_sModelTrPath;
            std::string m_sModelOnnxPath;
            std::string m_sEngineFp;
            std::string m_sEngineCache;

            MODE m_eMode;

            //neural network input dimension
            int m_iInput_h;
            int m_iInput_w;
            //original image width and height
            int m_iMrows;
            int m_iMcols;

            float m_fNmsThresh;
            float m_fDetectionThresh;

            //ONNX RUNTIME

            Ort::SessionOptions m_OrtSessionOptions;
            Ort::AllocatorWithDefaultOptions allocator;

            std::unique_ptr<Ort::Session> m_OrtSession;
            std::unique_ptr<Ort::Env> m_OrtEnv;

            //OnnxRuntime Input Model

            size_t num_input_nodes;
            std::vector<const char *> input_node_names;

            //OnnxRuntime Output Model

            size_t num_out_nodes;

            // onnx runtime data
            float *m_fpInputOnnxRuntime;
            float *m_fpOutOnnxRuntime[2];

            //Model Out
            std::vector<int64_t> m_viNumberOfBoundingBox;

            //Input dimension onnx model
            size_t m_InputTorchTensorSize;

            //LIBTORCH
            torch::Tensor m_TInputTorchTensor;
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

            //UTILS

            aiProductionReady::aiutils aut;

            //MESSAGE/ERROR HANDLING

            string m_sMessage;

            //FUNCTION

            //init

            void setOnnxRuntimeEnv();
            void setOnnxRuntimeModelInputOutput();
            void createYamlConfig(std::string modelPathOnnx, int input_h, int input_w, MODE t, std::string model_path);
            void setEnvVariable();
            void setSession();

            //Preprocessing
            //to preserve aspect ratio of image
            cv::Mat padding(cv::Mat &img, int width, int weight);

            //Accuracy

            //array with all detection accuracy
            Json::Value m_JsonRootArray;

            //use internally for detection accuracy
            cv::Rect get_RectMap(float bbox[4]);

        public:
            //string to save image id for accuracy detection
            string m_sAccurayImagePath;

            Yolov3();

            virtual ~Yolov3();

            bool init(std::string modelPathOnnx, int input_h, int input_w, MODE t, std::string model_path = NULL);

            void preprocessing(Mat &Image);
            torch::Tensor postprocessing();
            void runmodel();

            void createAccuracyFile();
            //return width of input image
            int getWidth()
            {

                return m_iInput_w;
            }

            //return height of input image
            int getHeight()
            {

                return m_iInput_h;
            }

            string getMessage()
            {

                return m_sMessage;
            }

            //call from outside class to draw rectangle given image
            cv::Rect get_rect(cv::Mat &img, float bbox[4]);

            //void nms(vector<float> &detection,float nsm_thres=0.9);

            float iou(float lbox[4], float rbox[4]);
        };

    } // namespace objectDetection

} //namespace aiProductionReady
