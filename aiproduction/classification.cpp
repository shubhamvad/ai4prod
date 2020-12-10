#include "classification.h"

#include "../deps/onnxruntime/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"
#include "../deps/onnxruntime/include/onnxruntime/core/providers/providers.h"

using namespace std;

using namespace onnxruntime;

namespace aiProductionReady
{
    namespace classification
    {

        ResNet50::ResNet50()
        {

            m_bInit = false;
            m_bCheckInit = false;
            m_bCheckPre = false;
            m_bCheckRun = false;
            m_bCheckPost = false;
        }

        void ResNet50::createYamlConfig()
        {

            //retrive or create config yaml file
            if (aut.checkFileExists(m_sModelTrPath + "/config.yaml"))
            {
                cout << "file1" << endl;

                m_ymlConfig = YAML::LoadFile(m_sModelTrPath + "/config.yaml");
            }
            else
            {

                // starts out as null
                m_ymlConfig["fp16"] = "0"; // it now is a map node
                m_ymlConfig["engine_cache"] = "1";
                m_ymlConfig["engine_path"] = m_sModelTrPath;
                std::ofstream fout(m_sModelTrPath + "/config.yaml");
                fout << m_ymlConfig;
            }
        }

        void ResNet50::setEnvVariable()
        {
            if (m_eMode == TensorRT)
            {
#ifdef __linux__

                string cacheModel = "ORT_TENSORRT_ENGINE_CACHE_ENABLE=" + m_ymlConfig["engine_cache"].as<std::string>();

                int cacheLenght = cacheModel.length();
                char cacheModelchar[cacheLenght + 1];
                strcpy(cacheModelchar, cacheModel.c_str());
                putenv(cacheModelchar);

                string fp16 = "ORT_TENSORRT_FP16_ENABLE=" + m_ymlConfig["fp16"].as<std::string>();
                int fp16Lenght = cacheModel.length();
                char fp16char[cacheLenght + 1];
                strcpy(fp16char, fp16.c_str());
                putenv(fp16char);

                m_sModelTrPath = "ORT_TENSORRT_ENGINE_CACHE_PATH=" + m_ymlConfig["engine_path"].as<std::string>();
                int n = m_sModelTrPath.length();
                char modelSavePath[n + 1];
                strcpy(modelSavePath, m_sModelTrPath.c_str());
                //esporto le path del modello di Tensorrt
                putenv(modelSavePath);

#elif _WIN32

                _putenv_s("ORT_TENSORRT_ENGINE_CACHE_ENABLE", m_ymlConfig["engine_cache"].as<std::string>().c_str());
                _putenv_s("ORT_TENSORRT_ENGINE_CACHE_PATH", m_ymlConfig["engine_path"].as<std::string>().c_str());
                _putenv_s("ORT_TENSORRT_FP16_ENABLE", m_ymlConfig["fp16"].as<std::string>().c_str());

#endif
            }
        }

        void ResNet50::setOnnxRuntimeEnv()
        {

            m_OrtEnv = std::make_unique<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "test"));

            if (m_eMode == Cpu)
            {

                m_OrtSessionOptions.SetIntraOpNumThreads(1);
                //ORT_ENABLE_ALL sembra avere le performance migliori
                m_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            }

            if (m_eMode == TensorRT)
            {

                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(m_OrtSessionOptions, 0));
            }
        }

        void ResNet50::setSession()
        {

#ifdef __linux__

            m_OrtSession = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv, m_sModelOnnxPath.c_str(), m_OrtSessionOptions));

#elif _WIN32

            //in windows devo inizializzarlo in questo modo
            std::wstring widestr = std::wstring(m_sModelOnnxPath.begin(), m_sModelOnnxPath.end());
            //session = new Ort::Session(*env, widestr.c_str(), m_OrtSessionOptions);
            m_OrtSession = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv, widestr.c_str(), m_OrtSessionOptions));

#endif
        }

        void ResNet50::setOnnxRuntimeModelInputOutput()
        {

            //INPUT
            num_input_nodes = m_OrtSession->GetInputCount();
            input_node_names = std::vector<const char *>(num_input_nodes);

            //OUTPUT
            num_out_nodes = m_OrtSession->GetOutputCount();
            out_node_names = std::vector<const char *>(num_out_nodes);
        }

        bool ResNet50::init(std::string modelPath, int width, int height, int ModelNumberOfClass, int NumberOfReturnedPrediction, MODE t, std::string modelTr_path)
        {
            if (!m_bCheckInit)
            {
                try
                {

                    m_sModelOnnxPath = modelPath;
                    //size at which image is resised
                    m_iInput_w = width;
                    m_iInput_h = height;
                    //by default input of neural network is 224 same as imagenet
                    m_iCropImage = 224;
                    m_iModelNumberOfClass = ModelNumberOfClass;
                    m_iNumberOfReturnedPrediction = NumberOfReturnedPrediction;
                    m_sModelTrPath = modelTr_path;
                    m_eMode = t;

                    createYamlConfig();

                    //set enviromental variable

                    setEnvVariable();

                    //OnnxRuntime set Env
                    setOnnxRuntimeEnv();

                    setSession();

                    //model input output
                    setOnnxRuntimeModelInputOutput();
                    m_bInit = true;
                    m_bCheckInit = true;
                    return true;
                }
                catch (const std::exception &e)
                {
                    std::cerr << e.what() << '\n';
                    m_bInit = true;
                    return false;
                }
            }
            else
            {
                cout << "Is not possibile to initialize more than one time" << endl;
            }
        }

        void ResNet50::preprocessing(Mat &Image)
        {

            //ResNet50::model=data;
            if (m_bInit && !m_bCheckPre && !m_bCheckRun && m_bCheckPost)
            {
                //resize(Image, Image, Size(256, 256), 0.5, 0.5, cv::INTER_LANCZOS4);
                resize(Image, Image, Size(m_iInput_h, m_iInput_w), 0, 0, cv::INTER_LINEAR);
                const int cropSize = m_iCropImage;
                const int offsetW = (Image.cols - cropSize) / 2;
                const int offsetH = (Image.rows - cropSize) / 2;
                const Rect roi(offsetW, offsetH, cropSize, cropSize);

                Image = Image(roi).clone();
                inputTensor = aut.convertMatToTensor(Image, Image.cols, Image.rows, Image.channels(), 1);

                //definisco la dimensione di input

                input_tensor_size = Image.cols * Image.rows * Image.channels();

                //Mat testImage;

                //testImage = convertTensortToMat(inputTensor, 224, 224);

                //imshow("test image", testImage);
                //imshow("original", Image);
                //waitKey(0);

                //verifico che immagine sia la stessa
                //equalImage(Image, testImage);

                //se le 2 immagini sono uguali allora noramlizzo il tensore
                //questi sono i valori di ImageNet

                inputTensor[0][0] = inputTensor[0][0].sub_(0.485).div_(0.229);
                inputTensor[0][1] = inputTensor[0][1].sub_(0.456).div_(0.224);
                inputTensor[0][2] = inputTensor[0][2].sub_(0.406).div_(0.225);

                m_bCheckPre = true;

                //cout << "preprocessing" << endl;

                //cout<<session->GetInputCount()<<endl;

                //cout<<session.GetInputCount()<<endl;

                //Preprocessig Image
            }
            else
            {

                cout << "call init() before" << endl;
            }
        }

        void ResNet50::runmodel()
        {

            //verifico che il tensore sia contiguous()

            if (inputTensor.is_contiguous() && m_bCheckPre)
            {

                //conversione del tensore a onnx runtime
                m_fpInOnnxRuntime = static_cast<float *>(inputTensor.storage().data());

                std::vector<float> input_tensor_values(input_tensor_size);

                for (int i = 0; i < input_tensor_size; i++)
                {

                    input_tensor_values[i] = m_fpInOnnxRuntime[i];
                }

                for (int i = 0; i < num_input_nodes; i++)
                {
                    // print input node names
                    char *input_name = m_OrtSession->GetInputName(i, allocator);
                    //printf("Input %d : name=%s\n", i, input_name);
                    input_node_names[i] = input_name;

                    // print input node types
                    Ort::TypeInfo type_info = m_OrtSession->GetInputTypeInfo(i);
                    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                    ONNXTensorElementDataType type = tensor_info.GetElementType();
                    //printf("Input %d : type=%d\n", i, type);

                    // print input shapes/dims
                    input_node_dims = tensor_info.GetShape();
                    //printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
                    //for (int j = 0; j < input_node_dims.size(); j++)
                    //printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
                }

                for (int i = 0; i < num_out_nodes; i++)
                {
                    // print input node names
                    char *input_name = m_OrtSession->GetOutputName(i, allocator);
                    //printf("Input %d : name=%s\n", i, input_name);

                    Ort::TypeInfo type_info = m_OrtSession->GetOutputTypeInfo(i);
                    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                    ONNXTensorElementDataType type = tensor_info.GetElementType();
                    //rintf("Input %d : type=%d\n", i, type);

                    // print input shapes/dims
                    out_node_dims = tensor_info.GetShape();
                    //printf("Input %d : num_dims=%zu\n", i, out_node_dims.size());
                    //for (int j = 0; j < out_node_dims.size(); j++)
                    //printf("Input %d : dim %d=%jd\n", i, j, out_node_dims[j]);
                }

                std::vector<const char *> output_node_names = {"output1"};

                // initialize input data with values in [0.0, 1.0]
                //for (unsigned int i = 0; i < input_tensor_size; i++)
                //    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

                // create input tensor object from data values
                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
                assert(input_tensor.IsTensor());

                auto output_tensors = m_OrtSession->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);

                assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

                assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

                m_fpOutOnnxRuntime = output_tensors.front().GetTensorMutableData<float>();

                float label;

                int cls;

                m_bCheckRun = true;
            }

            else
            {

                cout << "Cannot call run model without preprocessing" << endl;
            }
        }

        std::tuple<torch::Tensor, torch::Tensor> ResNet50::postprocessing()
        {   
            if(m_bCheckRun){

            //https://discuss.pytorch.org/t/can-i-initialize-tensor-from-std-vector-in-libtorch/33236/4
            m_TOutputTensor = torch::from_blob(m_fpOutOnnxRuntime, {m_iModelNumberOfClass}).clone();

            std::tuple<torch::Tensor, torch::Tensor> bestTopPrediction = torch::sort(m_TOutputTensor, 0, true);

            torch::Tensor indeces = torch::slice(std::get<1>(bestTopPrediction), 0, 0, m_iNumberOfReturnedPrediction, 1);
            torch::Tensor value = torch::slice(std::get<0>(bestTopPrediction), 0, 0, m_iNumberOfReturnedPrediction, 1);

            std::tuple<torch::Tensor, torch::Tensor> topPrediction = {indeces, value};

#ifdef EVAL_ACCURACY

            ofstream myfile;
            myfile.open("/home/aistudios/Desktop/classification-Detection.csv", std::ios::in | std::ios::out | std::ios::app);

            //remove file extension
            // size_t lastindex = m_sAccurayImagePath.find_last_of(".");
            // string imagePath_WithoutExt = m_sAccurayImagePath.substr(0, lastindex);

            //get only name file withot all path and extension
            std::string base_filename = m_sAccurayImagePath.substr(m_sAccurayImagePath.find_last_of("/\\") + 1);
            std::string::size_type const p(base_filename.find_last_of('.'));
            std::string file_without_extension = base_filename.substr(0, p);

            string stringToWrite = file_without_extension + "," + std::to_string(std::get<0>(topPrediction)[0].item<long>()) + "," + std::to_string(std::get<0>(topPrediction)[1].item<long>()) + "," + std::to_string(std::get<0>(topPrediction)[2].item<long>()) + "," + std::to_string(std::get<0>(topPrediction)[3].item<long>()) + "," + std::to_string(std::get<0>(topPrediction)[4].item<long>()) + "\n";

            myfile << stringToWrite.c_str();

            myfile.close();

#endif

            //this verify that you can only run pre run e post once for each new data
            m_bCheckRun=false;
            m_bCheckPre=false;
            m_bCheckPost=true;

            return topPrediction;

            }else{
                
                torch::Tensor m;
                torch::Tensor n;
                std::tuple<torch::Tensor, torch::Tensor> nullTensor = {n, m};
                cout<< "call run model before preporcessing"<<endl;
                return nullTensor;



            }
        }

        ResNet50::~ResNet50()
        {   //deallocate resources only if were allocated
            if (m_bCheckInit)
            {
                m_OrtSession.reset();
                m_OrtEnv.reset();
            }
        }

    } // namespace classification

} // namespace aiProductionReady