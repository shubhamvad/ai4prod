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

#include "objectdetection.h"

#ifdef TENSORRT

#include "../../deps/onnxruntime/tensorrt/include/onnxruntime/core/providers/providers.h"
#include "../../deps/onnxruntime/tensorrt/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"

#endif TENSORRT

#ifdef DIRECTML
#include "../../deps/onnxruntime/directml/include/onnxruntime/core/providers/providers.h"
#include "../../deps/onnxruntime/directml/include/onnxruntime/core/providers/dml/dml_provider_factory.h"

#endif


using namespace std::chrono;

namespace ai4prod
{
    namespace objectDetection
    {

        //costruttore di default
        Yolov3::Yolov3()
        {
            m_bInit = false;
            m_bCheckInit = false;
            m_bCheckPre = false;
            m_bCheckRun = false;
            m_bCheckPost = true;
        }

        void Yolov3::setOnnxRuntimeEnv()
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

        void Yolov3::setOnnxRuntimeModelInputOutput()
        {

            num_input_nodes = m_OrtSession->GetInputCount();
            input_node_names = std::vector<const char *>(num_input_nodes);

            num_out_nodes = m_OrtSession->GetOutputCount();
        }

        bool Yolov3::checkParameterConfig(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path)
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

        bool Yolov3::createYamlConfig(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path)
        {

            //retrive or create config yaml file
            if (aut.checkFileExists(model_path + "/config.yaml"))
            {

                m_ymlConfig = YAML::LoadFile(model_path + "/config.yaml");

                m_sEngineFp = m_ymlConfig["fp16"].as<std::string>();
                m_sEngineCache = m_ymlConfig["engine_cache"].as<std::string>();
                m_sModelTrPath = m_ymlConfig["engine_path"].as<std::string>();
                m_fNmsThresh = m_ymlConfig["Nms"].as<float>();
                m_fDetectionThresh = m_ymlConfig["DetectionThresh"].as<float>();
                m_iInput_w = m_ymlConfig["width"].as<int>();
                m_iInput_h = m_ymlConfig["height"].as<int>();
                m_eMode = aut.setMode(m_ymlConfig["Mode"].as<std::string>());
                m_sModelOnnxPath = m_ymlConfig["modelOnnxPath"].as<std::string>();
                m_iNumClasses = m_ymlConfig["numClasses"].as<int>();

                if (!checkParameterConfig(modelPathOnnx, input_h, input_w, numClasses, t, model_path))
                {
                    return false;
                }
                return true;
            }

            else
            {
                //first time parameter are initialized by default
                m_sEngineFp = "0";
                m_sEngineCache = "1";
                m_sModelTrPath = model_path;
                m_fNmsThresh = 0.5;
                m_fDetectionThresh = 0.01;
                m_iInput_w = input_w;
                m_iInput_h = input_h;
                m_eMode = t;
                m_sModelOnnxPath = modelPathOnnx;
                m_iNumClasses = numClasses;

                m_ymlConfig["fp16"] = m_sEngineFp;
                m_ymlConfig["engine_cache"] = m_sEngineCache;
                m_ymlConfig["engine_path"] = m_sModelTrPath;
                m_ymlConfig["Nms"] = m_fNmsThresh;
                m_ymlConfig["DetectionThresh"] = m_fDetectionThresh;
                m_ymlConfig["width"] = m_iInput_w;
                m_ymlConfig["height"] = m_iInput_h;
                m_ymlConfig["Mode"] = aut.setYamlMode(m_eMode);
                m_ymlConfig["modelOnnxPath"] = m_sModelOnnxPath;
                m_ymlConfig["numClasses"] = m_iNumClasses;

                std::ofstream fout(m_sModelTrPath + "/config.yaml");
                fout << m_ymlConfig;

                return true;
            }
        }

        void Yolov3::setEnvVariable()
        {

            if (m_eMode == TensorRT)
            {
#ifdef __linux__

                string cacheModel = "ORT_TENSORRT_ENGINE_CACHE_ENABLE=" + m_sEngineCache;

                int cacheLenght = cacheModel.length();
                char cacheModelchar[cacheLenght + 1];
                strcpy(cacheModelchar, cacheModel.c_str());
                putenv(cacheModelchar);

                string fp16 = "ORT_TENSORRT_FP16_ENABLE=" + m_sEngineFp;
                int fp16Lenght = cacheModel.length();
                char fp16char[cacheLenght + 1];
                strcpy(fp16char, fp16.c_str());
                putenv(fp16char);

                m_sModelTrPath = "ORT_TENSORRT_ENGINE_CACHE_PATH=" + m_sModelTrPath;
                int n = m_sModelTrPath.length();
                char modelSavePath[n + 1];
                strcpy(modelSavePath, m_sModelTrPath.c_str());
                //esporto le path del modello di Tensorrt
                putenv(modelSavePath);

#elif _WIN32

                _putenv_s("ORT_TENSORRT_ENGINE_CACHE_ENABLE", m_sEngineCache.c_str());
                _putenv_s("ORT_TENSORRT_ENGINE_CACHE_PATH", m_sModelTrPath.c_str());
                _putenv_s("ORT_TENSORRT_FP16_ENABLE", m_sEngineFp.c_str());

#endif
            }
        }

        void Yolov3::setSession()
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

        //function to initialize on Linux
        bool Yolov3::init(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path)
        {
            if (!m_bCheckInit)
            {
                try
                {

                    if (!aut.createFolderIfNotExist(model_path))
                    {

                        std::cout << "cannot create folder" << std::endl;

                        return false;
                    }

                    std::cout << "INIT MODE " << t << std::endl;

                    if (!createYamlConfig(modelPathOnnx, input_h, input_w, numClasses, t, model_path))
                    {

                        return false;
                    }

                    if (!aut.checkMode(m_eMode, m_sMessage))
                    {

                        std::cout << m_sMessage << std::endl;
                        return false;
                    }

                    //set enviromental variable

                    //setEnvVariable();

#ifdef __linux__

                    string cacheModel = "ORT_TENSORRT_ENGINE_CACHE_ENABLE=" + m_sEngineCache;

                    int cacheLenght = cacheModel.length();
                    char cacheModelchar[cacheLenght + 1];
                    strcpy(cacheModelchar, cacheModel.c_str());
                    putenv(cacheModelchar);

                    string fp16 = "ORT_TENSORRT_FP16_ENABLE=" + m_sEngineFp;
                    int fp16Lenght = cacheModel.length();
                    char fp16char[cacheLenght + 1];
                    strcpy(fp16char, fp16.c_str());
                    putenv(fp16char);

                    m_sModelTrPath = "ORT_TENSORRT_ENGINE_CACHE_PATH=" + m_sModelTrPath;
                    int n = m_sModelTrPath.length();
                    char modelSavePath[n + 1];
                    strcpy(modelSavePath, m_sModelTrPath.c_str());
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
                    return false;
                }
            }
            else
            {
                m_sMessage = "Is not possibile to call init() twice. Class already initialized";
                std::cout << "Is not possibile to call init() twice. Class already initialized" << std::endl;
            }
        }

        cv::Mat Yolov3::padding(cv::Mat &img, int width, int height)
        {

            int w, h, x, y;
            float r_w = width / (img.cols * 1.0);
            float r_h = height / (img.rows * 1.0);
            if (r_h > r_w)
            {
                w = width;
                h = r_w * img.rows;
                x = 0;
                y = (height - h) / 2;
            }
            else
            {
                w = r_h * img.cols;
                h = height;
                x = (width - w) / 2;
                y = 0;
            }
            cv::Mat re(h, w, CV_8UC3);
            cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
            cv::Mat out(height, width, CV_8UC3, cv::Scalar(128, 128, 128));
            re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
            return out;
        }

        void Yolov3::preprocessing(cv::Mat &Image)
        {
            if (m_bInit && !m_bCheckPre && !m_bCheckRun && m_bCheckPost)
            {

                m_iMcols = Image.cols;
                m_iMrows = Image.rows;

                //free all resources allocated

                m_viNumberOfBoundingBox.clear();

                Image = padding(Image, m_iInput_w, m_iInput_h);

                m_TInputTorchTensor = aut.convertMatToTensor(Image, Image.cols, Image.rows, Image.channels(), 1);

                m_InputTorchTensorSize = Image.cols * Image.rows * Image.channels();

                m_bCheckPre = true;
            }
            else
            {
                m_sMessage = "call init() before";
                std::cout << "call init() before" << std::endl;
            }
        }

        void Yolov3::runmodel()
        {

            if (m_TInputTorchTensor.is_contiguous() && m_bCheckPre)
            {
                //conversione del tensore a onnx runtime
                m_fpInputOnnxRuntime = static_cast<float *>(m_TInputTorchTensor.storage().data());

                std::vector<int64_t> input_node_dims;

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

                // for (int i = 0; i < num_out_nodes; i++)
                // {
                //     // print input node names
                //     char *input_name = m_OrtSession->GetOutputName(i, allocator);
                //     //printf("Output %d : name=%s\n", i, input_name);

                //     Ort::TypeInfo type_info = m_OrtSession->GetOutputTypeInfo(i);
                //     auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                //     ONNXTensorElementDataType type = tensor_info.GetElementType();
                //     //printf("Output %d : type=%d\n", i, type);

                //     // print input shapes/dims
                //     //out_node_dims = tensor_info.GetShape();
                //     //printf("Output %d : num_dims=%zu\n", i, out_node_dims.size());
                //     //for (int j = 0; j < out_node_dims.size(); j++)
                //     //printf("Output %d : dim %d=%jd\n", i, j, out_node_dims[j]);
                // }

                //https://github.com/microsoft/onnxruntime/issues/3170#issuecomment-596613449
                std::vector<const char *> output_node_names = {"classes", "boxes"};

                static const char *output_names[] = {"classes", "boxes"};
                static const size_t NUM_OUTPUTS = sizeof(output_names) / sizeof(output_names[0]);

                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, m_fpInputOnnxRuntime, m_InputTorchTensorSize, input_node_dims.data(), 4);

                assert(input_tensor.IsTensor());

                std::vector<Ort::Value> output_tensors = m_OrtSession->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);

                m_fpOutOnnxRuntime[0] = output_tensors[0].GetTensorMutableData<float>();
                m_fpOutOnnxRuntime[1] = output_tensors[1].GetTensorMutableData<float>();

                m_viNumberOfBoundingBox = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

                m_bCheckRun = true;
            }

            else
            {

                m_sMessage = "Cannot call run model without preprocessing";
                std::cout << "Cannot call run model without preprocessing" << std::endl;
            }
        } // namespace objectDetection

        torch::Tensor Yolov3::postprocessing()
        {
            if (m_bCheckRun)
            {

                //vettore contenente gli indici con soglia maggiore di 0.7
                std::vector<std::tuple<int, int, float>> bboxIndex;

                //vettore dei bounding box con soglia milgiore di 0.7
                std::vector<std::vector<float>> bboxValues;

                //m_viNumberOfBoundingBox[0]=(22743,80)
                //m_viNumberOfBoundingBox[1]=(22743,4)

                bool noDetection = true;
                bool noDetectionNms = true;

                auto start = high_resolution_clock::now();

                for (int index = 0; index < m_viNumberOfBoundingBox[0]; index++)
                {
                    float classProbability = 0.0;
                    int indexClassMaxValue = -1;
                    //numberofBoundingbox rappresenta il numero di classi di training
                    for (int classes = 0; classes < m_viNumberOfBoundingBox[1]; classes++)
                    {
                        //i*num_classes +j
                        //Detection threshold
                        if (m_fpOutOnnxRuntime[0][index * m_viNumberOfBoundingBox[1] + classes] > m_fDetectionThresh)
                        {
                            //serve per trovare il massimo per singolo bbox
                            if (m_fpOutOnnxRuntime[0][index * m_viNumberOfBoundingBox[1] + classes] > classProbability)
                            {

                                classProbability = m_fpOutOnnxRuntime[0][index * m_viNumberOfBoundingBox[1] + classes];
                                //aggiungo +1 perchè nel Map il valore delle classi parte da 1 e non da 0.Così come nel foglio dei nomi
                                indexClassMaxValue = classes + 1;
                            }
                        }
                    }

                    //inserisco elemento solo se è maggiore della soglia
                    if (indexClassMaxValue > -1)
                    {
                        //il primo valore è l'indice il secondo la classe
                        bboxIndex.push_back(std::make_tuple(index, indexClassMaxValue, classProbability));

                        // indice
                        float x = m_fpOutOnnxRuntime[1][index * 4];
                        float y = m_fpOutOnnxRuntime[1][index * 4 + 1];
                        float w = m_fpOutOnnxRuntime[1][index * 4 + 2];
                        float h = m_fpOutOnnxRuntime[1][index * 4 + 3];

                        std::vector<float> tmpbox{x, y, w, h, (float)indexClassMaxValue, classProbability};

                        bboxValues.push_back(tmpbox);

                        noDetection = false;
                    }
                }

                auto stop = high_resolution_clock::now();

                auto duration = duration_cast<microseconds>(stop - start);

                //cout << "SINGLE TIME INFERENCE 1 " << (double)duration.count() / (1000000) << "Sec" << endl;

                //no detection return empty tensor
                if (noDetection)
                {

                    //cout << "NO DETECTION " << noDetection << endl;
                    std::cout << "0Detection" << std::endl;
                    auto tensor = torch::ones({0});
                    m_bCheckRun = false;
                    m_bCheckPre = false;
                    m_bCheckPost = true;
                    return tensor;
                }

                else
                {

                    //NMS
                    auto start1 = high_resolution_clock::now();

                    std::vector<int> indexAfterNms;

                    for (int i = 0; i < bboxValues.size(); i++)
                    {

                        for (int j = i + 1; j < bboxValues.size(); j++)
                        {

                            if (std::get<1>(bboxIndex[i]) == std::get<1>(bboxIndex[j]))
                            {

                                //calcolo iou

                                float ar1[4] = {bboxValues[i][0], bboxValues[i][1], bboxValues[i][2], bboxValues[i][3]};
                                float ar2[4] = {bboxValues[j][0], bboxValues[j][1], bboxValues[j][2], bboxValues[j][3]};

                                float area1 = ((bboxValues[i][2] + bboxValues[i][0] - bboxValues[i][0]) * (bboxValues[i][3] + bboxValues[i][1] - bboxValues[i][1])) / 2;
                                float area2 = ((bboxValues[j][2] + bboxValues[j][0] - bboxValues[j][0]) * (bboxValues[j][3] + bboxValues[j][1] - bboxValues[j][1])) / 2;

                                float iouValue = iou(ar1, ar2);

                                //confronto intersezione bbox
                                if (iouValue > m_ymlConfig["Nms"].as<float>())
                                {

                                    if (std::get<2>(bboxIndex[i]) > std::get<2>(bboxIndex[j]))
                                    {
                                        //ritorno gli indici del vettore dei bounding box
                                        // indexAfterNms.push_back(std::get<0>(bboxIndex[i]));
                                        indexAfterNms.push_back(j);
                                        noDetectionNms = false;
                                    }
                                    else
                                    {

                                        indexAfterNms.push_back(i);
                                        noDetectionNms = false;
                                        break;
                                    }
                                }
                            }
                            //calcolo iou
                        }
                    }

                    auto stop1 = high_resolution_clock::now();

                    auto duration1 = duration_cast<microseconds>(stop1 - start1);

                    //cout << "SINGLE TIME INFERENCE 2 " << (double)duration1.count() / (1000000) << "Sec" << endl;

                    if (noDetectionNms)
                    {
                        //cout << "NO DETECTION " << noDetection << endl;
                        //cout << "0Detection" << endl;

                        auto tensor = torch::ones({0});
                        m_bCheckRun = false;
                        m_bCheckPre = false;
                        m_bCheckPost = true;

                        return tensor;
                    }

                    else
                    {

                        auto start2 = high_resolution_clock::now();

                        std::vector<std::vector<float>> bboxValuesNms;

                        for (int i = 0; i < bboxValues.size(); i++)
                        {

                            if (std::find(indexAfterNms.begin(), indexAfterNms.end(), i) != indexAfterNms.end())
                            {
                            }
                            else
                            {

                                bboxValuesNms.push_back(bboxValues[i]);
                            }
                        }

#ifdef EVAL_ACCURACY
                        //need for handling image path for COCO DATASET
                        //for every image

                        string image_id = m_sAccurayImagePath;

                        const size_t last_slash_idx = image_id.find_last_of("\\/");
                        if (std::string::npos != last_slash_idx)
                        {
                            image_id.erase(0, last_slash_idx + 1);
                        }

                        // Remove extension if present.
                        const size_t period_idx = image_id.rfind('.');
                        if (std::string::npos != period_idx)
                        {
                            image_id.erase(period_idx);
                        }

                        image_id.erase(0, image_id.find_first_not_of('0'));

                        //cout << image_id << endl;
                        cv::Rect brect;

                        for (int i = 0; i < bboxValuesNms.size(); i++)
                        {

                            Json::Value root;
                            // Json::Value categoryIdJson;
                            // Json::Value bboxJson;
                            // Json::Value score;
                            root["image_id"] = std::stoi(image_id);

                            int cocoCategory = 0;
                            //darknet has 80 class while coco has 90 classes. We need to handle different number of classes on output
                            //1
                            if ((int)bboxValuesNms[i][4] > 0 && (int)bboxValuesNms[i][4] <= 11)
                            {

                                cocoCategory = (int)bboxValuesNms[i][4];
                            }

                            //2
                            if ((int)bboxValuesNms[i][4] > 11 && (int)bboxValuesNms[i][4] <= 24)
                            {

                                cocoCategory = (int)bboxValuesNms[i][4] + 1;
                            }
                            //3
                            if ((int)bboxValuesNms[i][4] > 24 && (int)bboxValuesNms[i][4] <= 26)
                            {

                                cocoCategory = (int)bboxValuesNms[i][4] + 2;
                            }
                            //4
                            if ((int)bboxValuesNms[i][4] > 26 && (int)bboxValuesNms[i][4] <= 40)
                            {

                                cocoCategory = (int)bboxValuesNms[i][4] + 4;
                            }
                            //5
                            if ((int)bboxValuesNms[i][4] > 40 && (int)bboxValuesNms[i][4] <= 60)
                            {

                                cocoCategory = (int)bboxValuesNms[i][4] + 5;
                            }
                            //6
                            if ((int)bboxValuesNms[i][4] == 61)
                            {

                                cocoCategory = (int)bboxValuesNms[i][4] + 6;
                            }
                            //7
                            if ((int)bboxValuesNms[i][4] == 62)
                            {

                                cocoCategory = (int)bboxValuesNms[i][4] + 8;
                            }
                            //8
                            if ((int)bboxValuesNms[i][4] > 62 && (int)bboxValuesNms[i][4] <= 73)
                            {

                                cocoCategory = (int)bboxValuesNms[i][4] + 9;
                            }
                            //9
                            if ((int)bboxValuesNms[i][4] > 73 && (int)bboxValuesNms[i][4] <= 80)
                            {

                                cocoCategory = (int)bboxValuesNms[i][4] + 10;
                            }

                            root["category_id"] = cocoCategory;

                            Json::Value valueBBoxjson(Json::arrayValue);

                            float tmp[4] = {bboxValuesNms[i][0], bboxValuesNms[i][1], bboxValuesNms[i][2], bboxValuesNms[i][3]};

                            cv::Rect brect;
                            brect = get_RectMap(tmp);

                            valueBBoxjson.append(brect.x);
                            valueBBoxjson.append(brect.y);
                            valueBBoxjson.append(brect.width);
                            valueBBoxjson.append(brect.height);

                            root["bbox"] = valueBBoxjson;
                            root["score"] = bboxValuesNms[i][5];

                            m_JsonRootArray.append(root);
                        }

#endif

                        m_TOutputTensor = aut.convert2dVectorToTensor(bboxValuesNms);

                        //this verify that you can only run pre run e post once for each new data
                        m_bCheckRun = false;
                        m_bCheckPre = false;
                        m_bCheckPost = true;

                        auto stop2 = high_resolution_clock::now();

                        auto duration2 = duration_cast<microseconds>(stop2 - start2);

                        //cout << "SINGLE TIME INFERENCE 2 " << (double)duration2.count() / (1000000) << "Sec" << endl;

                        return m_TOutputTensor;
                    }
                }
            }
            else
            {

                torch::Tensor nullTensor;

                m_sMessage = "call run model before postprocessing";
                std::cout << "call run model before postprocessing" << std::endl;
                return nullTensor;
            }
        }

        //get rect coordinate in Yolo Format with padding as preprocessing

        cv::Rect Yolov3::get_rect(cv::Mat &img, float bbox[4])
        {
            int l, r, t, b;
            float r_w = m_iInput_w / (img.cols * 1.0);
            float r_h = m_iInput_h / (img.rows * 1.0);
            if (r_h > r_w)
            {
                l = bbox[0] - bbox[2] / 2.f;
                r = bbox[0] + bbox[2] / 2.f;
                t = bbox[1] - bbox[3] / 2.f - (m_iInput_h - r_w * img.rows) / 2;
                b = bbox[1] + bbox[3] / 2.f - (m_iInput_h - r_w * img.rows) / 2;
                l = l / r_w;
                r = r / r_w;
                t = t / r_w;
                b = b / r_w;
            }
            else
            {
                l = bbox[0] - bbox[2] / 2.f - (m_iInput_w - r_h * img.cols) / 2;
                r = bbox[0] + bbox[2] / 2.f - (m_iInput_w - r_h * img.cols) / 2;
                t = bbox[1] - bbox[3] / 2.f;
                b = bbox[1] + bbox[3] / 2.f;
                l = l / r_h;
                r = r / r_h;
                t = t / r_h;
                b = b / r_h;
            }
            return cv::Rect(l, t, r - l, b - t);
        }

        float Yolov3::iou(float lbox[4], float rbox[4])
        {

            float interBox[] = {
                std::max(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), //left
                std::min(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), //right
                std::max(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), //top
                std::min(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), //bottom
            };

            if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
                return 0.0f;

            float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
            return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
        }

        cv::Rect Yolov3::get_RectMap(float bbox[4])
        {

            int l, r, t, b;
            float r_w = m_iInput_w / (m_iMcols * 1.0);
            float r_h = m_iInput_h / (m_iMrows * 1.0);
            if (r_h > r_w)
            {
                l = bbox[0] - bbox[2] / 2.f;
                r = bbox[0] + bbox[2] / 2.f;
                t = bbox[1] - bbox[3] / 2.f - (m_iInput_h - r_w * m_iMrows) / 2;
                b = bbox[1] + bbox[3] / 2.f - (m_iInput_h - r_w * m_iMrows) / 2;
                l = l / r_w;
                r = r / r_w;
                t = t / r_w;
                b = b / r_w;
            }
            else
            {
                l = bbox[0] - bbox[2] / 2.f - (m_iInput_w - r_h * m_iMcols) / 2;
                r = bbox[0] + bbox[2] / 2.f - (m_iInput_w - r_h * m_iMcols) / 2;
                t = bbox[1] - bbox[3] / 2.f;
                b = bbox[1] + bbox[3] / 2.f;
                l = l / r_h;
                r = r / r_h;
                t = t / r_h;
                b = b / r_h;
            }
            return cv::Rect(l, t, r - l, b - t);
        }

        //we need to create this function because all detection need to be saved all together
        void Yolov3::createAccuracyFile()
        {

            Json::StreamWriterBuilder builder;
            const std::string json_file = Json::writeString(builder, m_JsonRootArray);
            //std::cout << json_file << std::endl;

            std::ofstream myfile;
            myfile.open("yoloVal.json", std::ios::in | std::ios::out | std::ios::app);
            myfile << json_file + "\n";
            myfile.close();
        }

        Yolov3::~Yolov3()
        {
            if (m_bCheckInit)
            {

                m_OrtSession.reset();
                m_OrtEnv.reset();
            }
        }

    } // namespace objectDetection
} // namespace ai4prod
