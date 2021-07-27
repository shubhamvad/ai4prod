/*

GNU GPL V3 License

Copyright (c) 2020 Eric Tondelli. All rights reserved.

This file is part of Ai4prod.

Ai4prod is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Ai4prod is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Ai4prod.  If not, see <http://www.gnu.org/licenses/>

*/

#include "classification.h"
#include "defines.h"


using namespace onnxruntime;

namespace ai4prod
{
    namespace classification
    {

        ResNet50::ResNet50()
        {

            m_bInit = false;
            m_bCheckInit = false;
            m_bCheckPre = false;
            m_bCheckRun = false;
            m_bCheckPost = true;
        }

        bool ResNet50::checkParameterConfig(std::string modelPathOnnx, int input_w, int input_h, int numClasses, MODE t)
        {
            if (m_eMode != t)
            {
                m_sMessage = "ERROR: mode initialization is different from configuration file. Please choose another save directory.";
                return false;
            }

            if (input_h != m_iInput_h)
            {
                m_sMessage = "ERROR: Image input height is different from configuration file.";
                return false;
            }

            if (input_w != m_iInput_w)
            {
                m_sMessage = "ERROR: Image input width is different from configuration file. ";
                return false;
            }

            if (m_iNumClasses != numClasses)
            {
                m_sMessage = "ERROR: Number of model class is different from configuration file.";

                return false;
            }

            if (m_eMode == TensorRT)
            {

                if (m_sModelOnnxPath != modelPathOnnx && m_sEngineCache == "1")
                {

                    m_sMessage = "WARNING: Use cache tensorrt engine file with different onnx Model";
                    return true;
                }
            }

            return true;
        }

        bool ResNet50::createYamlConfig(std::string modelPathOnnx, int input_w, int input_h, int numClasses, int NumberOfReturnedPrediction, MODE t, std::string modelTr_path)
        {

            //retrive or create config yaml file
            if (aut.checkFileExists(modelTr_path + "/config.yaml"))
            {

                m_ymlConfig = YAML::LoadFile(modelTr_path + "/config.yaml");

                m_sEngineFp = m_ymlConfig["fp16"].as<std::string>();
                m_sEngineCache = m_ymlConfig["engine_cache"].as<std::string>();
                m_sModelTrPath = m_ymlConfig["engine_path"].as<std::string>();
                m_iNumberOfReturnedPrediction = m_ymlConfig["outputClass"].as<int>();
                m_iNumClasses = m_ymlConfig["modelNumberOfClass"].as<int>();
                m_iInput_w = m_ymlConfig["width"].as<int>();
                m_iInput_h = m_ymlConfig["width"].as<int>();
                m_iCropImage = m_ymlConfig["crop"].as<int>();
                m_sModelOnnxPath = m_ymlConfig["modelOnnxPath"].as<std::string>();
                m_eMode = aut.setMode(m_ymlConfig["Mode"].as<std::string>());

                if (!checkParameterConfig(modelPathOnnx, input_w, input_h, numClasses, t))
                {
                    return false;
                }
                return true;
            }
            else
            {

                std::cout << "INIT" << std::endl;

                m_sModelOnnxPath = modelPathOnnx;
                //size at which image is resised
                m_iInput_w = input_w;
                m_iInput_h = input_h;
                //by default input of neural network is 224 same as imagenet
                m_iCropImage = 224;
                m_iNumClasses = numClasses;
                m_iNumberOfReturnedPrediction = NumberOfReturnedPrediction;
                m_sModelTrPath = modelTr_path;
                m_eMode = t;
                m_sEngineFp = "0";
                m_sEngineCache = "1";

                // starts out as null
                m_ymlConfig["fp16"] = m_sEngineFp; // it now is a map node
                m_ymlConfig["engine_cache"] = m_sEngineCache;
                m_ymlConfig["engine_path"] = m_sModelTrPath;
                m_ymlConfig["outputClass"] = m_iNumberOfReturnedPrediction;
                m_ymlConfig["modelNumberOfClass"] = m_iNumClasses;
                m_ymlConfig["width"] = m_iInput_w;
                m_ymlConfig["height"] = m_iInput_h;
                m_ymlConfig["crop"] = m_iCropImage;
                m_ymlConfig["modelOnnxPath"] = m_sModelOnnxPath;

                m_ymlConfig["Mode"] = aut.setYamlMode(m_eMode);

                std::ofstream fout(m_sModelTrPath + "/config.yaml");
                fout << m_ymlConfig;

                return true;
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

            std::cout << "MODE " << m_eMode << std::endl;

            if (m_eMode == TensorRT)
            {

#ifdef TENSORRT
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(m_OrtSessionOptions, 0));
#else
                std::cout << "Ai4prod not compiled with Tensorrt Execution Provider" << std::endl;
#endif
            }

            if (m_eMode == DirectML)
            {

#ifdef DIRECTML

                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(m_OrtSessionOptions, 0));

#else

                std::cout << "Ai4prod not compiled with DirectML Execution Provider" << std::endl;
#endif
            }
        }

        void ResNet50::setSession()
        {

#ifdef __linux__

            m_OrtSession = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv, m_sModelOnnxPath.c_str(), m_OrtSessionOptions));

#elif _WIN32

            //in windows devo inizializzarlo in questo modo

            std::cout << "MODEL PATH " << m_sModelOnnxPath.c_str() << std::endl;
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

        bool ResNet50::init(std::string modelPath, int width, int height, int numClasses, int NumberOfReturnedPrediction, MODE t, std::string modelTr_path)
        {
            if (!m_bCheckInit)
            {
                try
                {

                    if (!aut.createFolderIfNotExist(modelTr_path))
                    {

                        std::cout << "cannot create folder" << std::endl;

                        return false;
                    }

                    std::cout << "INIT MODE " << t << std::endl;

                    if (!createYamlConfig(modelPath, width, height, numClasses, NumberOfReturnedPrediction, t, modelTr_path))
                    {

                        std::cout << m_sMessage << std::endl;
                        return false;
                    };

                    if (!aut.checkMode(m_eMode, m_sMessage))
                    {

                        std::cout << m_sMessage << std::endl;
                        return false;
                    }

                    //ENV VARIABLE CANNOT BE SET INTO FUNCTION MUST BE ON MAIN THREAD

                    //I cannot set this code into a function and inside if
#ifdef __linux__

                    int ret;
                    std::string cacheModel = "ORT_TENSORRT_ENGINE_CACHE_ENABLE=" + m_sEngineCache;

                    int cacheLenght = cacheModel.length();
                    char cacheModelchar[cacheLenght + 1];
                    strcpy(cacheModelchar, cacheModel.c_str());
                    ret = putenv(cacheModelchar);

                    std::string fp16 = "ORT_TENSORRT_FP16_ENABLE=" + m_sEngineFp;
                    int fp16Lenght = cacheModel.length();
                    char fp16char[cacheLenght + 1];
                    strcpy(fp16char, fp16.c_str());
                    putenv(fp16char);
                    std::string modelTrTmp;

                    modelTrTmp = "ORT_TENSORRT_ENGINE_CACHE_PATH=" + m_sModelTrPath;
                    int n = modelTrTmp.length();
                    char modelSavePath[n + 1];
                    strcpy(modelSavePath, modelTrTmp.c_str());
                    //esporto le path del modello di Tensorrt

                    putenv(modelSavePath);

#elif _WIN32

                    _putenv_s("ORT_TENSORRT_ENGINE_CACHE_ENABLE", m_sEngineCache.c_str());
                    _putenv_s("ORT_TENSORRT_ENGINE_CACHE_PATH", m_sModelTrPath.c_str());
                    _putenv_s("ORT_TENSORRT_FP16_ENABLE", m_sEngineFp.c_str());

#endif

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
                m_sMessage = "Is not possibile to call init() twice. Class already initialized";
                std::cout << "Is not possibile to call init() twice. Class already initialized" << std::endl;
            }
        } // namespace classification

        void ResNet50::preprocessing(cv::Mat &Image)
        {
           
            //ResNet50::model=data;
            if (m_bInit && !m_bCheckPre && !m_bCheckRun && m_bCheckPost)
            {

                //resize(Image, Image, Size(256, 256), 0.5, 0.5, cv::INTER_LANCZOS4);
                resize(Image, Image, cv::Size(m_iInput_h, m_iInput_w), 0, 0, cv::INTER_LINEAR);
                const int cropSize = m_iCropImage;
                const int offsetW = (Image.cols - cropSize) / 2;
                const int offsetH = (Image.rows - cropSize) / 2;
                const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);

                Image = Image(roi).clone();
                inputTensor = aut.convertMatToTensor(Image, Image.cols, Image.rows, Image.channels(), 1);

                //define input size

                input_tensor_size = Image.cols * Image.rows * Image.channels();

                inputTensor[0][0] = inputTensor[0][0].sub_(0.485).div_(0.229);
                inputTensor[0][1] = inputTensor[0][1].sub_(0.456).div_(0.224);
                inputTensor[0][2] = inputTensor[0][2].sub_(0.406).div_(0.225);

                m_bCheckPre = true;
            }
            else
            {
                m_sMessage = "call init() before";
                std::cout << "call init() before" << std::endl;
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

                m_fpOutOnnxRuntime = output_tensors.front().GetTensorMutableData<float>();

                float label;

                int cls;

                m_bCheckRun = true;
            }

            else
            {
                m_sMessage = "Cannot call run model without preprocessing";
                std::cout << "Cannot call run model without preprocessing" << std::endl;
            }
        }

        std::tuple<torch::Tensor, torch::Tensor> ResNet50::postprocessing()
        {
            if (m_bCheckRun)
            {

                //https://discuss.pytorch.org/t/can-i-initialize-tensor-from-std-vector-in-libtorch/33236/4
                m_TOutputTensor = torch::from_blob(m_fpOutOnnxRuntime, {m_iNumClasses}).clone();

                std::tuple<torch::Tensor, torch::Tensor> bestTopPrediction = torch::sort(m_TOutputTensor, 0, true);

                torch::Tensor indeces = torch::slice(std::get<1>(bestTopPrediction), 0, 0, m_iNumberOfReturnedPrediction, 1);
                torch::Tensor value = torch::slice(std::get<0>(bestTopPrediction), 0, 0, m_iNumberOfReturnedPrediction, 1);

                std::tuple<torch::Tensor, torch::Tensor> topPrediction = {indeces, value};

#ifdef EVAL_ACCURACY

                std::cout << "EVAL_ACCURACY" << std::endl;
                std::ofstream myfile;
                myfile.open("classification-Detection.csv", std::ios::in | std::ios::out | std::ios::app);

                //remove file extension
                // size_t lastindex = m_sAccurayImagePath.find_last_of(".");
                // string imagePath_WithoutExt = m_sAccurayImagePath.substr(0, lastindex);

                //get only name file withot all path and extension
                std::string base_filename = m_sAccurayImagePath.substr(m_sAccurayImagePath.find_last_of("/\\") + 1);
                std::string::size_type const p(base_filename.find_last_of('.'));
                std::string file_without_extension = base_filename.substr(0, p);

                std::string stringToWrite = file_without_extension + "," + std::to_string(std::get<0>(topPrediction)[0].item<int>()) + "," + std::to_string(std::get<0>(topPrediction)[1].item<int>()) + "," + std::to_string(std::get<0>(topPrediction)[2].item<int>()) + "," + std::to_string(std::get<0>(topPrediction)[3].item<int>()) + "," + std::to_string(std::get<0>(topPrediction)[4].item<int>()) + "\n";

                myfile << stringToWrite.c_str();

                myfile.close();

#endif

                //this verify that you can only run pre run e post once for each new data
                m_bCheckRun = false;
                m_bCheckPre = false;
                m_bCheckPost = true;

                return topPrediction;
            }
            else
            {

                torch::Tensor m;
                torch::Tensor n;
                std::tuple<torch::Tensor, torch::Tensor> nullTensor = {n, m};
                m_sMessage = "call run model before postprocessing";
                std::cout << "call run model before postprocessing" << std::endl;
                return nullTensor;
            }
        }

        ResNet50::~ResNet50()
        { //deallocate resources only if were allocated
            if (m_bCheckInit)
            {
            m_OrtSession.release();
            m_OrtEnv.release();
            }
        }

    } // namespace classification

} // namespace ai4prod