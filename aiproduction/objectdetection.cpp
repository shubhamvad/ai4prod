#include "objectdetection.h"
#include "../deps/onnxruntime/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"
#include "../deps/onnxruntime/include/onnxruntime/core/providers/providers.h"

using namespace std;
namespace aiProductionReady
{
    namespace objectDetection
    {

        //costruttore di default
        Yolov3::Yolov3()
        {
        }

        Yolov3::Yolov3(std::string modelPathOnnx, int input_h, int input_w, MODE t, std::string modelTr_path)
        {

            //verifico se esiste il file di configurazione altrimenti ne creo uno

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

            //set width height of input image

            m_iInput_h = input_h;
            m_iInput_w = input_w;

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

            _putenv_s("ORT_TENSORRT_ENGINE_CACHE_ENABLE", "1");
            _putenv_s("ORT_TENSORRT_ENGINE_CACHE_PATH", modelTr_path.c_str());

#endif
            m_OrtEnv = std::make_unique<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "test"));

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

            m_OrtSession = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv, modelPathOnnx.c_str(), m_OrtSessionOptions));

#elif _WIN32

            //in windows devo inizializzarlo in questo modo
            std::wstring widestr = std::wstring(modelPathOnnx.begin(), modelPathOnnx.end());
            //session = new Ort::Session(*env, widestr.c_str(), m_OrtSessionOptions);
            m_OrtSession = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv, widestr.c_str(), m_OrtSessionOptions));

#endif

            //controlla quanti thread sono utilizzati

            //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0));

            //INPUT
            num_input_nodes = m_OrtSession->GetInputCount();
            input_node_names = std::vector<const char *>(num_input_nodes);

            //OUTPUT
            num_out_nodes = m_OrtSession->GetOutputCount();
            //out_node_names = std::vector<const char *>(num_out_nodes);

            cout << "sessione inizializzata" << endl;
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

        void Yolov3::preprocessing(Mat &Image)
        {

            //free all resources allocated

            m_viNumberOfBoundingBox.clear();

            Image = padding(Image, m_iInput_w, m_iInput_h);

            m_TInputTorchTensor = aut.convertMatToTensor(Image, Image.cols, Image.rows, Image.channels(), 1);

            m_InputTorchTensorSize = Image.cols * Image.rows * Image.channels();

            // imshow("insidePadding", Image);
            // waitKey(500);
            // cv::Mat test;
            // test = aut.convertTensortToMat(m_TInputTorchTensor, 608, 608);
        }

        void Yolov3::runmodel()
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

        } // namespace objectDetection

        torch::Tensor Yolov3::postprocessing()
        {

            //vettore contenente gli indici con soglia maggiore di 0.7
            std::vector<std::tuple<int, int, float>> bboxIndex;

            //vettore dei bounding box con soglia milgiore di 0.7
            vector<vector<float>> bboxValues;

            //m_viNumberOfBoundingBox[0]=(22743,80)
            //m_viNumberOfBoundingBox[1]=(22743,4)

            bool noDetection = true;
            bool noDetectionNms = true;

            for (int index = 0; index < m_viNumberOfBoundingBox[0]; index++)
            {
                float classProbability = 0.0;
                int indexClassMaxValue = -1;
                //numberofBoundingbox rappresenta il numero di classi di training
                for (int classes = 0; classes < m_viNumberOfBoundingBox[1]; classes++)
                {
                    //i*num_classes +j
                    if (m_fpOutOnnxRuntime[0][index * m_viNumberOfBoundingBox[1] + classes] > 0.5)
                    {
                        //serve per trovare il massimo per singolo bbox
                        if (m_fpOutOnnxRuntime[0][index * m_viNumberOfBoundingBox[1] + classes] > classProbability)
                        {

                            classProbability = m_fpOutOnnxRuntime[0][index * m_viNumberOfBoundingBox[1] + classes];
                            indexClassMaxValue = classes;
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

                    vector<float> tmpbox{x, y, w, h, (float)indexClassMaxValue, classProbability};

                    bboxValues.push_back(tmpbox);

                    noDetection = false;
                }
            }

            if (noDetection)
            {

                cout << "NO DETECTION " << noDetection << endl;
                cout << "0Detection" << endl;
                auto tensor = torch::ones({0});

                return tensor;
            }

            else
            {

                //NMS

                vector<int> indexAfterNms;

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
                            if (iouValue > 0.5)
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

                if (noDetectionNms)
                {
                    cout << "NO DETECTION " << noDetection << endl;
                    cout << "0Detection" << endl;

                    auto tensor = torch::ones({0});

                    return tensor;
                }

                else
                {

                    // for (int i = 0; i < indexAfterNms.size(); i++)
                    // {

                    //     //devo aggiungere -i perchè tutte le volte che elimino un elemento si riduce la dimensione
                    //     //dell'array quindi gli indici calcolati prima devono essere scalati
                    //     bboxValues.erase(bboxValues.begin() + indexAfterNms[i] - i);
                    // }

                    vector<vector<float>> bboxValuesNms;

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

                    // for (int i = 0; i < indexAfterNms.size(); i++)
                    // {

                    //     bboxValuesNms.push_back(bboxValues[indexAfterNms[i]]);
                    // }

#ifdef EVAL_ACCURACY

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

                    cout << image_id << endl;

                    //for every detection
                    Json::Value rootArray(Json::arrayValue);

                    for (int i = 0; i < bboxValues.size(); i++)
                    {

                        Json::Value root;
                        // Json::Value categoryIdJson;
                        // Json::Value bboxJson;
                        // Json::Value score;

                        Json::Value valueBBoxjson(Json::arrayValue);

                        valueBBoxjson.append(bboxValues[i][0]);
                        valueBBoxjson.append(bboxValues[i][1]);
                        valueBBoxjson.append(bboxValues[i][2]);
                        valueBBoxjson.append(bboxValues[i][3]);

                        root["image_id"] = image_id;
                        root["category_id"] = bboxValues[i][4];
                        root["bbox"] = valueBBoxjson;
                        root["score"] = bboxValues[i][5];

                        rootArray.append(root);
                    }

                    Json::StreamWriterBuilder builder;
                    const std::string json_file = Json::writeString(builder, rootArray);
                    std::cout << json_file << std::endl;

                    ofstream myfile;
                    myfile.open("yoloVal.json", std::ios::in | std::ios::out | std::ios::app);
                    myfile << json_file + "\n";
                    myfile.close();
                    
                    

#endif

                    // delete [] m_fpOutOnnxRuntime;
                    // delete m_fpInputOnnxRuntime;
                    // m_fpOutOnnxRuntime[0]=nullptr;
                    // m_fpOutOnnxRuntime[1]=nullptr;
                    // m_fpInputOnnxRuntime=nullptr;

                    cout << bboxValuesNms.size() << endl;

                    torch::Tensor Output = aut.convert2dVectorToTensor(bboxValuesNms);

                    return Output;
                }
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

        Yolov3::~Yolov3()
        {

            m_OrtSession.reset();
            m_OrtEnv.reset();
        }

    } // namespace objectDetection
} // namespace aiProductionReady
