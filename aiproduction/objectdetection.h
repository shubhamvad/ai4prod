#include "modelInterface.h"

namespace aiProductionReady{


namespace objectDetection
{

    class Yolov3 : aiProductionReady::modelInterfaceObjectDetection
    {

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
        size_t m_InputTorchTensorSize;

        float *m_fpInputOnnxRuntime;
        float *m_fpOutOnnxRuntime[2];

        torch::Tensor m_TInputTorchTensor;

        //path del modello di tensorrt
        std::string m_sModelTrPath;

        //funzioni Yolov3

        //dimensione immagine di input Yolo
        int m_iInput_h;
        int m_iInput_w;

        //dichiarando la funzione static essendo un membro privato posso scrivere la sua 
        //inizializzazione nel file .cpp
        //in fase di compilazione il compilatore ricompilerà solo il file cpp e non anche .h in caso di
        //di modifiche
        static cv::Mat padding(cv::Mat &img, int width, int weight);

        aiProductionReady::aiutils aut;

        //VARIABILI POST PROCESSING



        //funzioni di POST PROCESSING

        std::vector<int64_t> m_viNumberOfBoundingBox;

       

        //intersection over unit 
        //float iou(float lbox[4], float rbox[4]);

        //non max suppression
        //void nms(std::vector<Detection>& res, float *output, float nms_thresh = 0.9);

        

    public:
        Yolov3();

        virtual ~Yolov3();

        Yolov3(std::string modelPathOnnx, int input_h,int input_w,std::string modelTr_path = NULL);

        void preprocessing(Mat &Image);
        torch::Tensor postprocessing();
        void runmodel();

        //return width of input image
        int getWidth(){

            return m_iInput_w;
        }

        //return height of input image
        int getHeight(){

            return m_iInput_h;
        }


         //funzione inversa per il preprocessing
        cv::Rect get_rect(cv::Mat& img, float bbox[4]);

        //void nms(vector<float> &detection,float nsm_thres=0.9);

        float iou(float lbox[4], float rbox[4]);
    };

} // namespace objectDetection


}//namespace aiProductionReady

