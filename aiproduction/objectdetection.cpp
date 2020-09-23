
#include "objectdetection.h"
#include "../onnxruntime/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"
#include "../onnxruntime/include/onnxruntime/core/providers/providers.h"

using namespace std;
namespace aiProductionReady
{
    namespace objectDetection
    {

        //costruttore di default
        Yolov3::Yolov3()
        {
        }

        Yolov3::Yolov3(std::string modelPathOnnx, int input_h, int input_w, std::string modelTr_path)
        {

            //set width height of input image

            m_iInput_h = input_h;
            m_iInput_w = input_w;

            char cacheModel[] = "ORT_TENSORRT_ENGINE_CACHE_ENABLE=1";
            putenv(cacheModel);

            m_sModelTrPath = "ORT_TENSORRT_ENGINE_CACHE_PATH=" + modelTr_path;

            cout << m_sModelTrPath << endl;

            int n = m_sModelTrPath.length();
            char modelSavePath[n + 1];

            strcpy(modelSavePath, m_sModelTrPath.c_str());
            //esporto le path del modello di Tensorrt
            putenv(modelSavePath);

            env = new Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "test");

#ifdef CPU

            session_options.SetIntraOpNumThreads(1);
            //ORT_ENABLE_ALL sembra avere le performance migliori
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#endif

#ifdef TENSORRT

            //esporto le variabili
            m_sModelTrPath = modelTr_path;

            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0));
#endif

            session = new Ort::Session(*env, modelPathOnnx.c_str(), session_options);

            //controlla quanti thread sono utilizzati

            //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0));

            //INPUT
            num_input_nodes = session->GetInputCount();
            input_node_names = std::vector<const char *>(num_input_nodes);

            //OUTPUT
            num_out_nodes = session->GetOutputCount();
            out_node_names = std::vector<const char *>(num_out_nodes);

            cout << "sessione inizializzata" << endl;
        }

        Yolov3::~Yolov3()
        {
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

        torch::Tensor Yolov3::preprocessing(Mat &Image)
        {
            cv::Mat padImage;

            padImage = padding(Image, m_iInput_w, m_iInput_h);

            inputTensor = aut.convertMatToTensor(padImage, padImage.cols, padImage.rows, padImage.channels(), 1);

            input_tensor_size = Image.cols * Image.rows * Image.channels();

            //Normalizzazione
            // inputTensor[0][0] = inputTensor[0][0].sub_(0.485).div_(0.229);
            // inputTensor[0][1] = inputTensor[0][1].sub_(0.456).div_(0.224);
            // inputTensor[0][2] = inputTensor[0][2].sub_(0.406).div_(0.225);

            //inputTensor[0][0]=inputTensor[0][0].div(255);
            //inputTensor[0][1]=inputTensor[0][1].div(255);
            //inputTensor[0][2]=inputTensor[0][2].div(255);

            cv::Mat test;
            test = aut.convertTensortToMat(inputTensor, 608, 608);

            imshow("test", test);

            waitKey(0);
            return inputTensor;
        }

        torch::Tensor Yolov3::runmodel(torch::Tensor &input)
        {

            //conversione del tensore a onnx runtime
            p = static_cast<float *>(inputTensor.storage().data());

            std::vector<float> input_tensor_values(input_tensor_size);

            for (int i = 0; i < input_tensor_size; i++)
            {

                input_tensor_values[i] = p[i];
            }

            for (int i = 0; i < num_input_nodes; i++)
            {
                // print input node names
                char *input_name = session->GetInputName(i, allocator);
                printf("Input %d : name=%s\n", i, input_name);
                input_node_names[i] = input_name;

                // print input node types
                Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                ONNXTensorElementDataType type = tensor_info.GetElementType();
                printf("Input %d : type=%d\n", i, type);

                // print input shapes/dims
                input_node_dims = tensor_info.GetShape();
                printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
                for (int j = 0; j < input_node_dims.size(); j++)
                    printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
            }

            for (int i = 0; i < num_out_nodes; i++)
            {
                // print input node names
                char *input_name = session->GetOutputName(i, allocator);
                printf("Output %d : name=%s\n", i, input_name);

                Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                ONNXTensorElementDataType type = tensor_info.GetElementType();
                printf("Output %d : type=%d\n", i, type);

                // print input shapes/dims
                out_node_dims = tensor_info.GetShape();
                printf("Output %d : num_dims=%zu\n", i, out_node_dims.size());
                for (int j = 0; j < out_node_dims.size(); j++)
                    printf("Output %d : dim %d=%jd\n", i, j, out_node_dims[j]);
            }

            //https://github.com/microsoft/onnxruntime/issues/3170#issuecomment-596613449
            std::vector<const char *> output_node_names = {"classes", "boxes"};

            static const char *output_names[] = {"classes", "boxes"};
            static const size_t NUM_OUTPUTS = sizeof(output_names) / sizeof(output_names[0]);

            OrtValue *p_output_tensors[NUM_OUTPUTS] = {nullptr};

            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
            assert(input_tensor.IsTensor());

            clock_t start, end;

            cout << input_node_names[0] << endl;

            start = clock();
            std::vector<Ort::Value> output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
            end = clock();

            //assert(output_tensors.size() == 2 && output_tensors.front().IsTensor());

            // for (const auto &dim : output_tensors[0].GetTensorTypeAndShapeInfo().GetShape())
            //     std::cout << dim << " ";
            std::cout << std::endl;

            double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
            cout << "Time taken by program is : " << fixed
                 << time_taken << setprecision(5);
            cout << " sec " << endl;

            cout << "tensorSize" << output_tensors.size() << endl;

            //cout<<output_tensors[1].GetCount()<<endl;

            float *cls[2];

            float *test;
            //Ort::Value::GetTensorMutableData(output_tensors[0],(void**)test);

            float *classes = nullptr;
            //output_tensors[0].GetTensorMutableData<float>();

            //OrtApi ort;

            //ort.GetTensorMutableData(output_tensors[0],(void**)&classes);

            //float *boxes=output_tensors[1].GetTensorMutableData<float>();

            cls[0] = output_tensors[0].GetTensorMutableData<float>();
            cls[1] = output_tensors[1].GetTensorMutableData<float>();

            float *arr = output_tensors[0].GetTensorMutableData<float>();

            //cout << output_tensors[0].GetStringTensorDataLength() << endl;
            cout << output_tensors[0].GetTensorTypeAndShapeInfo().GetShape() << endl;

            cout << output_tensors[1].GetTensorTypeAndShapeInfo().GetShape() << endl;

            //numero totale dei bounding box (19*19 +38*38 +76*76 )*3 con rete 608
            auto numberOfBoundingBox = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

            cout << numberOfBoundingBox << endl;

            //vettore contenente gli indici con soglia maggiore di 0.7
            std::vector<std::tuple<int, int, float>> bboxIndex;

            //vettore dei bounding box con soglia milgiore di 0.7
            vector<vector<float>> bboxValues;

            // for (int i = 0; i < 90972; i++)
            // {
            //     cout<<cls[1][i]<<endl;
            //     int tt;
            //     cin>>tt;
            //     if (cls[1][i] == 457.32867)
            //     {

            //         cout <<"BBOX INDEX "<< i << endl;
            //     }
            // }

            for (int i = 0; i < numberOfBoundingBox[0]; i++)
            {
                float max = 0.0;
                int index = 0;
                //numberofBoundingbox rappresenta il numero di classi di training
                for (int j = 0; j < numberOfBoundingBox[1]; j++)
                {

                    if (cls[0][i * numberOfBoundingBox[1] + j] > 0.7)
                    {
                        //serve per trovare il massimo per singolo bbox
                        if (cls[0][i * numberOfBoundingBox[1] + j] > max)
                        {

                            max = cls[0][i * numberOfBoundingBox[1] + j];
                            index = j;
                        }
                    }
                }

                //inserisco elemento solo se è maggiore della soglia
                if (index > 0)
                {
                    //il primo valore è l'indice il secondo la classe
                    bboxIndex.push_back(std::make_tuple(i, index, max));
                    cout << "INDEX " << i << endl;
                    float x = cls[1][i * 4];
                    float y = cls[1][i * 4 + 1];
                    float w = cls[1][i * 4 + 2];
                    float h = cls[1][i * 4 + 3];

                    vector<float> tmpbox{x, y, w, h};

                    bboxValues.push_back(tmpbox);
                }
            }

            //NMS

            vector<int> indexAfterNms;

            cout << "SIZE VALUE" << bboxValues.size() << endl;

            for (int i = 0; i < bboxValues.size(); i++)
            {

                for (int j = i + 1; j < bboxValues.size(); j++)
                {

                    //verifico se hanno la stessa classe

                    //cout<<std::get<1>(bboxIndex[i])<<endl;
                    //cout<<std::get<1>(bboxIndex[j])<<endl;
                    //cout<<"INDEX j"<<j<<endl;

                    if (std::get<1>(bboxIndex[i]) == std::get<1>(bboxIndex[j]))
                    {

                        //calcolo iou
                        cout << bboxValues[i][0] << endl;
                        float ar1[4] = {bboxValues[i][0], bboxValues[i][1], bboxValues[i][2], bboxValues[i][3]};
                        float ar2[4] = {bboxValues[j][0], bboxValues[j][1], bboxValues[j][2], bboxValues[j][3]};

                        float area1=((bboxValues[i][2]+bboxValues[i][0]-bboxValues[i][0]) * (bboxValues[i][3]+bboxValues[i][1]-bboxValues[i][1]))/2;
                        float area2=((bboxValues[j][2]+bboxValues[j][0]-bboxValues[j][0]) * (bboxValues[j][3]+bboxValues[j][1]-bboxValues[j][1]))/2;

                        cout<<"AREA 1"<<area1<<endl;
                        cout<<"AREA 2"<<area2<<endl;

                        float iouValue = iou(ar1, ar2);
                        
                        cout<<"IOUVALUE"<<iouValue<<endl;
                        //confronto intersezione bbox
                        if (iouValue > 0.6)
                        {
                            // cout<<"NUMERO CLASSE"<<std::get<1>(bboxIndex[i])<<endl;
                            // cout<<std::get<1>(bboxIndex[j])<<endl;

                            // cout << "CONFIDENZA CLASSI " << std::get<2>(bboxIndex[i]) << endl;
                            // cout << std::get<2>(bboxIndex[j]) << endl;

                            //prendo l'indice del valore della classe da eliminare
                            if (std::get<2>(bboxIndex[i]) > std::get<2>(bboxIndex[j]))
                            {
                                //ritorno gli indici del vettore dei bounding box
                                // indexAfterNms.push_back(std::get<0>(bboxIndex[i]));
                                indexAfterNms.push_back(j);
                            }
                            else
                            {

                                indexAfterNms.push_back(i);
                                break;
                            }

                            // if (area1>area2){

                            //      indexAfterNms.push_back(j);

                            // }else{

                            //     indexAfterNms.push_back(i);
                            //     break;

                            // }



                        }
                    }
                    //calcolo iou
                }
            }

            cout<<"BBOX VALUE "<<bboxValues<<endl;

            cout<<"INDEX AFTER NMS "<<indexAfterNms<<endl;
            //elimino

            for (int i = 0; i < indexAfterNms.size(); i++)
            {
                cout<< "INDEX ERASE "<<indexAfterNms[i]<<endl;
                //devo aggiungere -i perchè tutte le volte che elimino un elemento si riduce la dimensione 
                //dell'array quindi gli indici calcolati prima devono essere scalati
                bboxValues.erase(bboxValues.begin() + indexAfterNms[i]-i);
            }


            Mat img;
            img = imread("/home/nvidia/Develop/aiproductionready/test/objectDetection/dog.jpg");

            int indNms = 0;
            for (auto &value : bboxValues)
            {

                cv::Rect brect;
                cout << value << endl;

                float tmp[4] = {value[0], value[1], value[2], value[3]};

                brect = get_rect(img, tmp);

                cv::rectangle(img, brect, cv::Scalar(0, 255, 0));
            }

            imshow("immagine", img);
            waitKey(0);
            // clock_t start2, end2;
            // start2 = clock();
            // auto output_tensors2 = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
            // end2 = clock();

            // double time_taken2 = double(end2 - start2) / double(CLOCKS_PER_SEC);
            // cout << "Time taken by program is : " << fixed
            //      << time_taken2 << setprecision(5);
            // cout << " sec " << endl;
        } // namespace objectDetection

        torch::Tensor Yolov3::postprocessing(torch::Tensor &input)
        {



        }


        

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

        // void nms(vector<float> &detection, float nsm_thres)
        // {
        //     for (auto value : detection)
        //     {
        //     }
        // }

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

    } // namespace objectDetection

} // namespace aiProductionReady