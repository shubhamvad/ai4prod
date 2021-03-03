#include "instancesegmentation.h"
#include "../../deps/onnxruntime/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"
#include "../../deps/onnxruntime/include/onnxruntime/core/providers/providers.h"

namespace ai4prod
{
    namespace instanceSegmentation
    {

        Yolact::Yolact()
        {
        }

        void Yolact::createYamlConfig(std::string modelPathOnnx, int input_h, int input_w, MODE t, std::string model_path)
        {

            //retrive or create config yaml file
            if (m_aut.checkFileExists(model_path + "/config.yaml"))
            {

                m_ymlConfig = YAML::LoadFile(model_path + "/config.yaml");

                m_sEngineFp = m_ymlConfig["fp16"].as<std::string>();
                m_sEngineCache = m_ymlConfig["engine_cache"].as<std::string>();
                m_sModelTrPath = m_ymlConfig["engine_path"].as<std::string>();
                m_fNmsThresh = m_ymlConfig["Nms"].as<float>();
                m_fDetectionThresh = m_ymlConfig["DetectionThresh"].as<float>();
                m_iInput_w = m_ymlConfig["width"].as<int>();
                m_iInput_h = m_ymlConfig["height"].as<int>();
                m_eMode = m_aut.setMode(m_ymlConfig["Mode"].as<std::string>());
                m_sModelOnnxPath = m_ymlConfig["modelOnnxPath"].as<std::string>();
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

                m_ymlConfig["fp16"] = m_sEngineFp;
                m_ymlConfig["engine_cache"] = m_sEngineCache;
                m_ymlConfig["engine_path"] = m_sModelTrPath;
                m_ymlConfig["Nms"] = m_fNmsThresh;
                m_ymlConfig["DetectionThresh"] = m_fDetectionThresh;
                m_ymlConfig["width"] = m_iInput_w;
                m_ymlConfig["height"] = m_iInput_h;
                m_ymlConfig["Mode"] = m_aut.setYamlMode(m_eMode);
                m_ymlConfig["modelOnnxPath"] = m_sModelOnnxPath;

                std::ofstream fout(m_sModelTrPath + "/config.yaml");
                fout << m_ymlConfig;
            }
        }
        void Yolact::setOnnxRuntimeEnv()
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

        void Yolact::setSession()
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
        void Yolact::setOnnxRuntimeModelInputOutput()
        {
            m_num_input_nodes = m_OrtSession->GetInputCount();
            m_input_node_names = std::vector<const char *>(m_num_input_nodes);

            m_num_out_nodes = m_OrtSession->GetOutputCount();

            m_output_node_names = std::vector<const char *>(m_num_out_nodes);
        }
        bool Yolact::init(std::string modelPathOnnx, int input_h, int input_w, MODE t, std::string model_path)
        {

            if (!m_aut.createFolderIfNotExist(model_path))
            {

                cout << "cannot create folder" << endl;

                return false;
            }

            cout << "INIT MODE " << t << endl;

            createYamlConfig(modelPathOnnx, input_h, input_w, t, model_path);

            if (!m_aut.checkMode(m_eMode, m_sMessage))
            {

                cout << m_sMessage << endl;
                return false;
            }

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

            setOnnxRuntimeEnv();

            setSession();

            setOnnxRuntimeModelInputOutput();
        }

        void Yolact::preprocessing(Mat &Image)
        {
            Mat tmpImage;
            tmpImage = Image.clone();

            testImage = Image.clone();

            //set original image dimension
            m_ImageHeightOrig = Image.rows;
            m_ImageWidhtOrig = Image.cols;

            //cv::imshow("test", tmpImage);

            //tensor with RGB channel
            m_TInputTensor = m_aut.convertMatToTensor8bit(tmpImage, tmpImage.cols, tmpImage.rows, tmpImage.channels(), 1);

            // auto imgAfter = m_aut.convertTensortToMat8bit(m_TInputTensor, Image.cols, Image.rows);

            // cv::imshow("test2", imgAfter);
            // cv::waitKey(0);

            //torch::nn::Functional::InterpolateFuncOptions().scale_factor({ 500 }).mode(torch::kBilinear).align_corners(false);

            cout << "AFTER" << endl;

            m_TInputTensor = torch::nn::functional::interpolate(m_TInputTensor, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{550, 550}).mode(torch::kBilinear).align_corners(false));

            auto size = m_TInputTensor.sizes();

            m_InputTorchTensorSize = size[1] * size[2] * size[3];

            cout << "valore DImensione" << m_InputTorchTensorSize << endl;
            //normalization pixel image are in range [0,255]

            m_TInputTensor[0][0] = m_TInputTensor[0][0].sub_(123.8).div_(58.40);
            m_TInputTensor[0][1] = m_TInputTensor[0][1].sub_(116.78).div_(57.12);
            m_TInputTensor[0][2] = m_TInputTensor[0][2].sub_(103.94).div_(57.38);

            // for (int i = 0; i < 200; i++)
            // {

            //     cout << m_TInputTensor[0][0][0][i] << endl;
            // }
        }

        void Yolact::runmodel()
        {

            if (m_TInputTensor.is_contiguous())
            {
                m_TInputTensor = m_TInputTensor;

                m_fpInputOnnxRuntime = static_cast<float *>(m_TInputTensor.storage().data());

                std::vector<int64_t> input_node_dims;
                std::vector<int64_t> output_node_dims;

                //set variable input onnxruntime
                for (int i = 0; i < m_num_input_nodes; i++)
                {
                    //get input node name
                    char *input_name = m_OrtSession->GetInputName(i, allocator);
                    printf("Output %d : name=%s\n", i, input_name);
                    m_input_node_names[i] = input_name;

                    //save input node dimension
                    Ort::TypeInfo type_info = m_OrtSession->GetInputTypeInfo(i);

                    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                    input_node_dims = tensor_info.GetShape();
                }

                //set variable output onnxruntime

                for (int i = 0; i < m_num_out_nodes; i++)
                {

                    //get input node name
                    char *output_name = m_OrtSession->GetOutputName(i, allocator);

                    printf("Output %d : name=%s\n", i, output_name);
                    //m_input_node_names[i] = output_name;
                    m_output_node_names[i] = output_name;
                    //save input node dimension
                    Ort::TypeInfo type_info = m_OrtSession->GetOutputTypeInfo(i);
                    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                    output_node_dims = tensor_info.GetShape();
                }

                //Session Inference

                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, m_fpInputOnnxRuntime, m_InputTorchTensorSize, input_node_dims.data(), 4);

                auto tensortData = input_tensor.GetTensorMutableData<float>();

                cout << "INPUT DIMENSION " << input_node_dims << endl;
                cout << "INPUT SIZW DIMENSION " << input_tensor.GetTensorTypeAndShapeInfo().GetShape() << endl;

                auto tensorINput_test = torch::from_blob((float *)(tensortData), {1, 3, 550, 550}).clone();

                // for (int k = 0; k < 500; k++)
                // {

                //     cout << tensorINput_test[0][0][0][k].item<float>()+0.002 << endl;
                // }

                cout << "output name " << m_output_node_names << endl;
                assert(input_tensor.IsTensor());

                std::vector<Ort::Value> output_tensors = m_OrtSession->Run(Ort::RunOptions{nullptr}, m_input_node_names.data(), &input_tensor, 1, m_output_node_names.data(), 5);

                //persist data across function

                m_fpOutOnnxRuntime[0] = output_tensors[0].GetTensorMutableData<float>();
                m_fpOutOnnxRuntime[1] = output_tensors[1].GetTensorMutableData<float>();
                m_fpOutOnnxRuntime[2] = output_tensors[2].GetTensorMutableData<float>();
                m_fpOutOnnxRuntime[3] = output_tensors[3].GetTensorMutableData<float>();
                m_fpOutOnnxRuntime[4] = output_tensors[4].GetTensorMutableData<float>();

                cout << "loc " << output_tensors[0].GetTensorTypeAndShapeInfo().GetShape() << endl;
                cout << " conf " << output_tensors[1].GetTensorTypeAndShapeInfo().GetShape() << endl;
                cout << " mask " << output_tensors[2].GetTensorTypeAndShapeInfo().GetShape() << endl;
                cout << " priors " << output_tensors[3].GetTensorTypeAndShapeInfo().GetShape() << endl;
                cout << " proto " << output_tensors[4].GetTensorTypeAndShapeInfo().GetShape() << endl;
            }
            else
            {

                cout << "error" << endl;
            }
        }

        torch::Tensor Yolact::postprocessing()
        {
            auto locTensor = torch::from_blob((float *)(m_fpOutOnnxRuntime[0]), {1, 19248, 4}).clone();
            auto confTensor = torch::from_blob((float *)(m_fpOutOnnxRuntime[1]), {1, 19248, 81}).clone();
            auto maskTensor = torch::from_blob((float *)(m_fpOutOnnxRuntime[2]), {1, 19248, 32}).clone();
            auto priorsTensor = torch::from_blob((float *)(m_fpOutOnnxRuntime[3]), {19248, 4}).clone();
            auto protoTensor = torch::from_blob((float *)(m_fpOutOnnxRuntime[4]), {1, 138, 138, 32}).clone();

            // cout<< "LOC TENSOR SIZE "<< locTensor.sizes()<<endl;
            // cout << "TEST TENSOR" <<locTensor[0][1][1]<<endl;

            cout << "Onnxrutime Value " << m_fpOutOnnxRuntime[0][1] << endl;

            //tensor comparison between libtorch onnxruntime PRINT DATA
            for (int i = 0; i < 4; i++)
            {

                cout << m_fpOutOnnxRuntime[0][i] << endl;
            }

            for (int i = 0; i < 4; i++)
            {

                cout << locTensor[0][i][0].item<float>() << endl;
                cout << locTensor[0][i][1].item<float>() << endl;
                cout << locTensor[0][i][2].item<float>() << endl;
                cout << locTensor[0][i][3].item<float>() << endl;
            }

            int batch_size = locTensor.sizes()[0];
            int num_priors = priorsTensor.sizes()[0];

            cout << "NUM CLASSES " << num_priors << " " << batch_size << endl;

            auto confPreds = confTensor.view({batch_size, num_priors, 81}).transpose(2, 1).contiguous();

            //decode Function

            float variances[2] = {0.1, 0.2};

            auto cat1 = priorsTensor.index({torch::indexing::Slice(None),
                                            torch::indexing::Slice(None, 2)})
                            .contiguous();

            // 0 is the value of tensor in batch size
            // not good for multiple batch size
            auto cat2 = locTensor[0].index({torch::indexing::Slice(None),
                                            torch::indexing::Slice(None, 2)})
                            .contiguous();

            auto cat3 = priorsTensor.index({torch::indexing::Slice(None),
                                            torch::indexing::Slice(2, None)})
                            .contiguous();

            auto cat4 = locTensor[0].index({torch::indexing::Slice(None),
                                            torch::indexing::Slice(2, None)})
                            .contiguous();

            auto finalExp = torch::exp(cat4 * variances[1]);

            auto catFinal = cat1 + cat2 * variances[0] * cat3;

            auto catFinal2 = cat3 * finalExp;

            auto decoded_boxes = torch::cat({catFinal, catFinal2}, 1);

            cout << "box size DECODE " << decoded_boxes.sizes()[0] << endl;

            for (int i = 0; i < decoded_boxes.sizes()[0]; i++)
            {

                decoded_boxes[i][0] = decoded_boxes[i][0] - (decoded_boxes[i][2] / 2);
                decoded_boxes[i][1] = decoded_boxes[i][1] - (decoded_boxes[i][3] / 2);

                decoded_boxes[i][2] = decoded_boxes[i][2] + (decoded_boxes[i][0]);
                decoded_boxes[i][3] = decoded_boxes[i][3] + (decoded_boxes[i][1]);
            }

            //detect function

            cout << confPreds.sizes() << endl;

            auto cur_scores = confPreds[0];

            cur_scores = cur_scores.index({{torch::indexing::Slice(1, None),
                                            torch::indexing::Slice(None)}});

            //this is a tuple
            auto [conf_scores, conf_index] = torch::max(cur_scores, 0);

            //0.05 detection threshold
            torch::Tensor keep = {conf_scores > 0.05};

            cout << "KEEP SIZE 1" << endl;

            // for(int i=0;i<100;i++){

            //     cout<< keep[i].item<float>()<<endl;

            // }

            auto scores = cur_scores.index({torch::indexing::Slice(None),
                                            keep});

            auto boxes = decoded_boxes.index({keep, torch::indexing::Slice(None)});
            //[0] is the batch size element. If one element is 0
            auto mask = maskTensor[0].index({keep, torch::indexing::Slice(None)});

            //No tensor Found

            if (scores.sizes()[1] == 0)
            {

                auto tensor = torch::ones({0});
                return tensor;
            }

            //fast_nms------------------------------

            int topk = 200;
            auto [scores_nms, idx_nms] = scores.sort(1, true);

            idx_nms = idx_nms.index({torch::indexing::Slice(None),
                                     torch::indexing::Slice(None, topk)})
                          .contiguous();

            scores = scores.index({torch::indexing::Slice(None),
                                   torch::indexing::Slice(None, topk)})
                         .contiguous();

            cout << "SCORES NMS" << endl;

            // for (int i; i<43;i++){

            //     cout <<scores_nms[0][i].item<float>()<<endl;
            // }

            //Gli score_nms sono uguali

            int num_classes = idx_nms.sizes()[0];
            int num_dets = idx_nms.sizes()[1];

            cout << "NUM CLASSES " << num_classes << " " << num_dets << " " << endl;

            auto boxes_nms = boxes.index({idx_nms.view(-1),
                                          torch::indexing::Slice(None)});

            boxes_nms = boxes_nms.view({num_classes, num_dets, 4}).contiguous();

            auto mask_nms = mask.index({idx_nms.view(-1),
                                        torch::indexing::Slice(None)});

            mask_nms = mask_nms.view({num_classes, num_dets, -1}).contiguous();

            cout << "BOX NmS PRE " << boxes_nms.sizes() << endl;
            //jaccard------------------------------------

            //if (boxes_nms.dim()==2){
            cout << "box Size" << endl;

            //auto box_a = boxes_nms.index({torch::indexing::Slice(None), "..."}).clone();
            //auto box_b = boxes_nms.index({torch::indexing::Slice(None), "..."}).clone();

            auto box_a = boxes_nms.clone();
            auto box_b = boxes_nms.clone();
            //}

            int n = box_a.sizes()[0];
            int A = box_a.sizes()[1];
            int B = box_b.sizes()[1];

            //intersection calculation-----------------------------------------
            cout << box_a.sizes() << endl;
            cout << "box A " << box_a[0][0] << endl;
            cout << "box B" << box_b[0][1] << endl;
            cout << "n " << n << endl;
            cout << "A " << A << endl;
            cout << "B " << B << endl;
            //min

            // auto torch_test = torch::min(box_a[0][0], box_b[0][1]);

            // cout << "TEST " << torch_test << endl;

            //FINO A QUI box_A e Box_b sono uguali al Python

            auto box_a_min = box_a.index({torch::indexing::Slice(None),
                                          torch::indexing::Slice(None),
                                          torch::indexing::Slice(2, None)})
                                 .unsqueeze(2)
                                 .expand({n, A, B, 2})
                                 .clone();

            auto box_b_min = box_b.index({torch::indexing::Slice(None),
                                          torch::indexing::Slice(None),
                                          torch::indexing::Slice(2, None)})
                                 .unsqueeze(1)
                                 .expand({n, A, B, 2})
                                 .clone();

            auto max_xy = torch::min(box_a_min, box_b_min);

            //max

            auto box_a_max = box_a.index({torch::indexing::Slice(None),
                                          torch::indexing::Slice(None),
                                          torch::indexing::Slice(None, 2)})
                                 .unsqueeze(2)
                                 .expand({n, A, B, 2})
                                 .clone();

            auto box_b_max = box_b.index({torch::indexing::Slice(None),
                                          torch::indexing::Slice(None),
                                          torch::indexing::Slice(None, 2)})
                                 .unsqueeze(1)
                                 .expand({n, A, B, 2})
                                 .clone();

            cout << "BOX A MIN " << box_a_min.sizes() << endl;
            auto min_xy = torch::max(box_a_max, box_b_max);

            auto inter = torch::clamp((max_xy - min_xy), 0);

            cout << "0.5" << endl;
            auto inter1 = inter.index({torch::indexing::Slice(None), torch::indexing::Slice(None),
                                       torch::indexing::Slice(None), 0});

            auto inter2 = inter.index({torch::indexing::Slice(None), torch::indexing::Slice(None),
                                       torch::indexing::Slice(None), 1});

            // cout << "Inter 0" << inter1[0][0] << endl;

            // cout << "Inter 2" << inter2[0][0] << endl;

            auto intersect = inter1 * inter2;

            //intersection calculation-------------------------------------------

            cout << "1" << endl;
            auto area_a = ((box_a.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 2}) -
                            box_a.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 0})) *
                           (box_a.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 3}) -
                            box_a.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 1})))
                              .unsqueeze(2)
                              .expand_as(intersect);

            cout << "2" << endl;

            auto area_b = ((box_b.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 2}) -
                            box_b.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 0})) *
                           (box_b.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 3}) -
                            box_b.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 1})))
                              .unsqueeze(2)
                              .expand_as(intersect);

            cout << "3" << endl;

            torch::Tensor areaUnion = area_a + area_b - intersect;

            //cout << "Area Union " <<areaUnion[0]<<endl;

            //cout << "INtersect"  << intersect[0]<< endl;

            torch::Tensor iou = intersect / areaUnion;

            //cout << "max IOU " <<iou[0]<<endl;

            //jaccard -------------------------------------
            //put all 0 under the main diagonal matrix
            iou = iou.triu(1);

            cout << "4" << endl;
            auto [iou_max, indeces_max] = torch::max(iou, 1);

            //cout << "max IOU " <<iou_max<<endl;

            //0.5 iou_threshold
            torch::Tensor keep_iou = iou_max < 0.51;

            //cout<<"IOU VALUE KEEP"<<keep_iou<<endl;

            cout << "KEEP_IOU SIZE " << keep_iou.sizes() << endl;
            cout << "IOU SIZE " << iou.sizes() << endl;

            cout << "5" << endl;

            cout << "box nms " << boxes_nms.sizes() << endl;

            cout << "mask nms " << mask_nms.sizes() << endl;
            cout << "scores nms " << scores_nms.sizes() << endl;

            auto classes = torch::arange(num_classes).index({torch::indexing::Slice(None), None}).expand_as(keep_iou);

            classes = classes.index({keep_iou});

            boxes_nms = boxes_nms.index({keep_iou});
            mask_nms = mask_nms.index({keep_iou});
            scores_nms = scores_nms.index({keep_iou});

            cout << "box nms " << boxes_nms.sizes() << endl;

            cout << "mask nms " << mask_nms.sizes() << endl;
            cout << "scores nms " << scores_nms.sizes() << endl;

            auto [final_scores, idx] = torch::sort(scores_nms, 0, true);

            //200 max num of detection
            idx = idx.index({torch::indexing::Slice(None, 100)});

            final_scores = final_scores.index({torch::indexing::Slice(None, 100)});

            classes = classes.index(idx);
            boxes_nms = boxes_nms.index(idx);
            mask_nms = mask_nms.index(idx);

            //--------------------------------postprocess Python

            //all score above threshold
            auto final_keep = (final_scores > 0.51);

            auto final_classes = classes.index(final_keep);
            auto final_boxes = boxes_nms.index(final_keep);
            auto final_mask_nms = mask_nms.index(final_keep);

            auto x = final_boxes.index({torch::indexing::Slice(None), 0});
            auto y = final_boxes.index({torch::indexing::Slice(None), 1});
            auto width = final_boxes.index({torch::indexing::Slice(None), 2});
            auto height = final_boxes.index({torch::indexing::Slice(None), 3});

            auto x1 = x * m_ImageWidhtOrig;
            auto y1 = y * m_ImageHeightOrig;
            auto width1 = width * m_ImageWidhtOrig;
            auto height1 = height * m_ImageHeightOrig;

            auto bbox_x = torch::min(x1, width1);
            auto bbox_width = torch::max(x1, width1);

            auto bbox_y = torch::min(y1, height1);
            auto bbox_height = torch::max(y1, height1);

            bbox_x = torch::clamp(bbox_x, 0);
            bbox_y = torch::clamp(bbox_y, 0);

            bbox_width = torch::clamp(bbox_width,0 ,m_ImageWidhtOrig);
            bbox_height = torch::clamp(bbox_height,0 ,m_ImageHeightOrig);

            cout<<"BOX "<<bbox_x<<endl;
            cout<<"BOX "<<bbox_y<<endl;
            cout<<"BOX "<<bbox_width<<endl;
            cout<<"BOX "<<bbox_height<<endl;

            cout<<"Width "<<m_ImageWidhtOrig<<endl;
            cout<<"Height "<<m_ImageHeightOrig<<endl;
            

            for (int i = 0; i < bbox_x.sizes()[0]; i++)
            {

                int x_rect = bbox_x[i].item<int>();
                int y_rect = bbox_y[i].item<int>();
                int width_rect = bbox_width[i].item<int>() -bbox_x[i].item<int>() ;
                int height_rect = bbox_height[i].item<int>()- bbox_y[i].item<int>();

                Rect rect(x_rect, y_rect, width_rect, height_rect);

                rectangle(testImage, rect, (255, 255, 255), 0.5);
            }
            
            cout<<"BOX "<<final_boxes<<endl;

            imshow("final Image",testImage);
            waitKey(0);

            
            cout << "final classes " << final_classes << endl;

            cout << "final boxes " << final_boxes.sizes() << endl;

            cout << "final mask_nms " << final_mask_nms.sizes() << endl;

            cout << "box_nms " << boxes_nms.sizes() << endl;
            cout << "mask_nms " << mask_nms.sizes() << endl;
            cout << "scores_nms " << final_scores.sizes() << endl;
            cout << "classes_nms " << classes.sizes() << endl;

            cout << "iou out" << iou.sizes() << endl;
            cout << "inters ect " << intersect.sizes() << endl;

            cout << "box_a " << box_a.sizes() << endl;
            cout << "box b " << box_b.sizes() << endl;

            cout << "box_nms" << boxes_nms.sizes() << endl;

            cout << "boxes Size" << boxes.sizes() << endl;
            cout << "Mask Size " << mask.sizes() << endl;
            cout << "index Scores True" << scores.sizes() << endl;
            //cout << "tensor Keep" << keep.sizes() << " Value " << keep[0] << endl;
            cout << "max Tensor Size " << conf_scores.sizes() << endl;

            auto tensor = torch::ones({0});
            return tensor;
        }

        Yolact::~Yolact()
        {

            m_OrtSession.reset();
            m_OrtEnv.reset();
        }

    } // namespace instanceSegmentation

} // namespace ai4prod