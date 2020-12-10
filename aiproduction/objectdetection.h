#include "modelInterface.h"

namespace aiProductionReady
{

    namespace objectDetection
    {

        class AIPRODUCTION_EXPORT Yolov3 : aiProductionReady::modelInterfaceObjectDetection
        {

        private:
            //ONNX RUNTIME
            Ort::SessionOptions m_OrtSessionOptions;
            std::unique_ptr<Ort::Session> m_OrtSession;
            std::unique_ptr<Ort::Env> m_OrtEnv;
            Ort::AllocatorWithDefaultOptions allocator;

            //OnnxRuntime Input Model

            size_t num_input_nodes;
            std::vector<const char *> input_node_names;

            //OnnxRuntime Output Model

            size_t num_out_nodes;

            //onnx model tensor input size
            size_t m_InputTorchTensorSize;

            float *m_fpInputOnnxRuntime;
            float *m_fpOutOnnxRuntime[2];

            torch::Tensor m_TInputTorchTensor;

            //INIT Function

            void setOnnxRuntimeEnv();
            void setOnnxRuntimeModelInputOutput();
            void createYamlConfig();
            void setEnvVariable();
            void setSession();

            //INIT Variable

            YAML::Node m_ymlConfig;
            string m_sModelOnnxPath;
            MODE m_eMode;
            std::string m_sModelTrPath;
            int m_iInput_h;
            int m_iInput_w;

         

            //Preprocessing

            cv::Mat padding(cv::Mat &img, int width, int weight);

            aiProductionReady::aiutils aut;

            //Post processing
            std::vector<int64_t> m_viNumberOfBoundingBox;

            //Accuracy

            //array with all detection accuracy
            Json::Value m_JsonRootArray;

            int m_iMrows;
            int m_iMcols;
            cv::Rect get_RectMap(float bbox[4]);

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

            //funzione inversa per il preprocessing
            cv::Rect get_rect(cv::Mat &img, float bbox[4]);

            //void nms(vector<float> &detection,float nsm_thres=0.9);

            float iou(float lbox[4], float rbox[4]);
        };

    } // namespace objectDetection

} //namespace aiProductionReady
