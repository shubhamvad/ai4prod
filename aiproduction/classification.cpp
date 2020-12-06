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

            //inizializzazione sessione OnnxRuntime
            //const char * model_path= "/home/tondelli/Desktop/2020/aiproductionready/onnxruntime/model/cpu/squeezenet.onnx";

            //Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "test");
            //session = new Ort::Session(env, model_path, session_options);

            //session_options.SetIntraOpNumThreads(6);
            //session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        }

        ResNet50::ResNet50(std::string path, int ModelNumberOfClass, int NumberOfReturnedPrediction, MODE t, std::string modelTr_path)
        {   

             if (aut.checkFileExists(modelTr_path + "/config.yaml"))
            {
                cout << "file1" << endl;

                m_ymlConfig = YAML::LoadFile(modelTr_path + "/config.yaml");
            }
            else
            {

                // starts out as null
                m_ymlConfig["fp16"] = "0"; // it now is a map node
                m_ymlConfig["engine_cache"] = "1";
                m_ymlConfig["engine_path"] = modelTr_path;
                std::ofstream fout(modelTr_path + "/config.yaml");
                fout << m_ymlConfig;
            }


            m_iModelNumberOfClass = ModelNumberOfClass;
            m_iNumberOfReturnedPrediction = NumberOfReturnedPrediction;

            //queste variabili devono essere settate prima di inzializzara la sessione
            //string cacheModel = "ORT_TENSORRT_ENGINE_CACHE_ENABLE=" + m_ymlConfig["engine_cache"].as<std::string>();

            //m_sModelTrPath = "ORT_TENSORRT_ENGINE_CACHE_PATH=" + modelTr_path;

            //cout << m_sModelTrPath << endl;

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

            //test = OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options,0);
#elif _WIN32

            _putenv_s("ORT_TENSORRT_ENGINE_CACHE_ENABLE", m_ymlConfig["engine_cache"].as<std::string>().c_str());
            _putenv_s("ORT_TENSORRT_ENGINE_CACHE_PATH", m_ymlConfig["engine_path"].as<std::string>().c_str());
            _putenv_s("ORT_TENSORRT_FP16_ENABLE", m_ymlConfig["fp16"].as<std::string>().c_str());

#endif
            m_OrtEnv = std::make_unique<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "test"));

            //option must be set before session initialization
            if (t == Cpu)
            {

                m_OrtSessionOptions.SetIntraOpNumThreads(1);
                //ORT_ENABLE_ALL sembra avere le performance migliori
                m_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            }

            if (t == TensorRT)
            {

                //esporto le variabili
                m_sModelTrPath = modelTr_path;

                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(m_OrtSessionOptions, 0));
            }

#ifdef __linux__

            m_OrtSession = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv, path.c_str(), m_OrtSessionOptions));

#elif _WIN32

            //in windows devo inizializzarlo in questo modo
            std::wstring widestr = std::wstring(path.begin(), path.end());
            //m_OrtSession = new Ort::Session(*env, widestr.c_str(), m_OrtSessionOptions);
            m_OrtSession = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv, widestr.c_str(), m_OrtSessionOptions));

#endif

            //INPUT
            num_input_nodes = m_OrtSession->GetInputCount();
            input_node_names = std::vector<const char *>(num_input_nodes);

            //OUTPUT
            num_out_nodes = m_OrtSession->GetOutputCount();
            out_node_names = std::vector<const char *>(num_out_nodes);

            //cout << "sessione inizializzata" << endl;
        }

        void ResNet50::preprocessing(Mat &Image)
        {

            //ResNet50::model=data;


            //resize(Image, Image, Size(256, 256), 0.5, 0.5, cv::INTER_LANCZOS4);
            resize(Image, Image, Size(256, 256), 0, 0, cv::INTER_LINEAR);
            const int cropSize = 224;
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

            //cout << "preprocessing" << endl;

            //cout<<session->GetInputCount()<<endl;

            //cout<<session.GetInputCount()<<endl;

            //Preprocessig Image
        }

        void ResNet50::runmodel()
        {

            //verifico che il tensore sia contiguous()

            if (inputTensor.is_contiguous())
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
            }

            else
            {

                cout << "Il tensore non è contiguous non possibile eseguire inferenza" << endl;
            }
        }

        std::tuple<torch::Tensor, torch::Tensor> ResNet50::postprocessing()
        {

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

            return topPrediction;
        }

        ResNet50::~ResNet50()
        {
            m_OrtSession.reset();
            m_OrtEnv.reset();
        }

    } // namespace classification

} // namespace aiProductionReady