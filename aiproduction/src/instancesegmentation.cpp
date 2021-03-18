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

        //verify if all parameters are good.
        //for example if Mode: tensorRt is the same as initialization
        bool Yolact::checkParameterConfig(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path)
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

            if(m_eMode==TensorRT){
                
                
                if (m_sModelOnnxPath!=modelPathOnnx && m_sEngineCache=="1"){

                m_sMessage = "WARNING: Use cache tensorrt engine file with different onnx Model";
                return true;
                }
            }

            return true;
        }

        //create config file if not present
        //return false if something is not configured correctly
        bool Yolact::createYamlConfig(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path)
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
                m_ymlConfig["Mode"] = m_aut.setYamlMode(m_eMode);
                m_ymlConfig["modelOnnxPath"] = m_sModelOnnxPath;
                m_ymlConfig["numClasses"] = m_iNumClasses;

                std::ofstream fout(m_sModelTrPath + "/config.yaml");
                fout << m_ymlConfig;

                return true;
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
        bool Yolact::init(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path)
        {

            if (!m_aut.createFolderIfNotExist(model_path))
            {

                cout << "cannot create folder" << endl;

                return false;
            }

            cout << "INIT MODE " << t << endl;
            //create config file and check for configuration error
            if (!createYamlConfig(modelPathOnnx, input_h, input_w, numClasses, t, model_path))
            {

                cout << m_sMessage << endl;
                return false;
            }

            //verify if Mode is implemented
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

            //set original image dimension
            m_ImageHeightOrig = Image.rows;
            m_ImageWidhtOrig = Image.cols;

            //tensor with RGB channel
            m_TInputTensor = m_aut.convertMatToTensor8bit(tmpImage, tmpImage.cols, tmpImage.rows, tmpImage.channels(), 1);

            m_TInputTensor = torch::nn::functional::interpolate(m_TInputTensor, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{550, 550}).mode(torch::kBilinear).align_corners(false));

            auto size = m_TInputTensor.sizes();

            m_InputTorchTensorSize = size[1] * size[2] * size[3];

            m_TInputTensor[0][0] = m_TInputTensor[0][0].sub_(123.8).div_(58.40);
            m_TInputTensor[0][1] = m_TInputTensor[0][1].sub_(116.78).div_(57.12);
            m_TInputTensor[0][2] = m_TInputTensor[0][2].sub_(103.94).div_(57.38);
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
                    //printf("Output %d : name=%s\n", i, input_name);
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

                    //printf("Output %d : name=%s\n", i, output_name);
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

                auto tensorINput_test = torch::from_blob((float *)(tensortData), {1, 3, 550, 550}).clone();

                assert(input_tensor.IsTensor());

                std::vector<Ort::Value> output_tensors = m_OrtSession->Run(Ort::RunOptions{nullptr}, m_input_node_names.data(), &input_tensor, 1, m_output_node_names.data(), 5);

                //persist data across function

                m_fpOutOnnxRuntime[0] = output_tensors[0].GetTensorMutableData<float>();
                m_fpOutOnnxRuntime[1] = output_tensors[1].GetTensorMutableData<float>();
                m_fpOutOnnxRuntime[2] = output_tensors[2].GetTensorMutableData<float>();
                m_fpOutOnnxRuntime[3] = output_tensors[3].GetTensorMutableData<float>();
                m_fpOutOnnxRuntime[4] = output_tensors[4].GetTensorMutableData<float>();
            }
            else
            {

                //cout << "error" << endl;
            }
        }

        torch::Tensor Yolact::decode(torch::Tensor &locTensor, torch::Tensor &priorsTensor, int batchSizePos)
        {

            float variances[2] = {0.1, 0.2};

            auto cat1 = priorsTensor.index({torch::indexing::Slice(None),
                                            torch::indexing::Slice(None, 2)})
                            .contiguous();

            auto cat2 = locTensor[batchSizePos].index({torch::indexing::Slice(None),
                                                       torch::indexing::Slice(None, 2)})
                            .contiguous();

            auto cat3 = priorsTensor.index({torch::indexing::Slice(None),
                                            torch::indexing::Slice(2, None)})
                            .contiguous();

            auto cat4 = locTensor[batchSizePos].index({torch::indexing::Slice(None),
                                                       torch::indexing::Slice(2, None)})
                            .contiguous();

            auto finalExp = torch::exp(cat4 * variances[1]);
            auto catFinal = cat1 + cat2 * variances[0] * cat3;
            auto catFinal2 = cat3 * finalExp;
            auto decoded_boxes = torch::cat({catFinal, catFinal2}, 1);

            decoded_boxes.index({torch::indexing::Slice(None), torch::indexing::Slice(None, 2)}) -= decoded_boxes.index({torch::indexing::Slice(None), torch::indexing::Slice(2, None)}) / 2;
            decoded_boxes.index({torch::indexing::Slice(None), torch::indexing::Slice(2, None)}) += decoded_boxes.index({torch::indexing::Slice(None), torch::indexing::Slice(None, 2)});

            return decoded_boxes;
        }
        /*
        Calculate intersection value for all bbox applying min max 
        */
        torch::Tensor Yolact::intersect(torch::Tensor box_a, torch::Tensor box_b)
        {

            int n = box_a.sizes()[0];
            int A = box_a.sizes()[1];
            int B = box_b.sizes()[1];

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

            auto min_xy = torch::max(box_a_max, box_b_max);

            auto inter = torch::clamp((max_xy - min_xy), 0);

            auto inter1 = inter.index({torch::indexing::Slice(None), torch::indexing::Slice(None),
                                       torch::indexing::Slice(None), 0});

            auto inter2 = inter.index({torch::indexing::Slice(None), torch::indexing::Slice(None),
                                       torch::indexing::Slice(None), 1});

            torch::Tensor intersect = inter1 * inter2;

            return intersect;
        }

        /*
        calculate intersection over union as intersection/union 
        */
        torch::Tensor Yolact::jaccard(torch::Tensor boxes_a, torch::Tensor boxes_b)
        {

            auto box_a = boxes_a.clone();
            auto box_b = boxes_b.clone();

            torch::Tensor inter = intersect(box_a, box_b);

            auto area_a = ((box_a.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 2}) -
                            box_a.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 0})) *
                           (box_a.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 3}) -
                            box_a.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 1})))
                              .unsqueeze(2)
                              .expand_as(inter);

            auto area_b = ((box_b.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 2}) -
                            box_b.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 0})) *
                           (box_b.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 3}) -
                            box_b.index({torch::indexing::Slice(None),
                                         torch::indexing::Slice(None), 1})))
                              .unsqueeze(2)
                              .expand_as(inter);

            torch::Tensor areaUnion = area_a + area_b - inter;

            torch::Tensor iou = inter / areaUnion;

            return iou;
        }

        /*
        Fast NMS as reported in paper Yolact
        */
        void Yolact::FastNms(InstanceSegmentationResult &result, float nms_thres, int topk)
        {

            auto [scores_nms, idx_nms] = result.scores.sort(1, true);

            //scores=scores_nms;

            idx_nms = idx_nms.index({torch::indexing::Slice(None),
                                     torch::indexing::Slice(None, topk)})
                          .contiguous();

            result.scores = scores_nms.index({torch::indexing::Slice(None),
                                              torch::indexing::Slice(None, topk)})
                                .contiguous();

            int num_classes = idx_nms.sizes()[0];
            int num_dets = idx_nms.sizes()[1];

            result.boxes = result.boxes.index({idx_nms.view(-1),
                                               torch::indexing::Slice(None)});

            result.boxes = result.boxes.view({num_classes, num_dets, 4}).contiguous();

            result.masks = result.masks.index({idx_nms.view(-1),
                                               torch::indexing::Slice(None)});

            result.masks = result.masks.view({num_classes, num_dets, -1}).contiguous();

            torch::Tensor iou = jaccard(result.boxes, result.boxes);

            iou = iou.triu(1);

            auto [iou_max, indeces_max] = torch::max(iou, 1);

            //0.5 iou_threshold
            torch::Tensor keep_iou = iou_max <= m_fNmsThresh;

            result.classes = torch::arange(num_classes).index({torch::indexing::Slice(None), None}).expand_as(keep_iou);

            result.classes = result.classes.index({keep_iou});

            result.boxes = result.boxes.index({keep_iou});
            result.masks = result.masks.index({keep_iou});
            result.scores = result.scores.index({keep_iou});

            auto [final_scores, idx] = torch::sort(result.scores, 0, true);

            //200 max num of detection
            idx = idx.index({torch::indexing::Slice(None, 100)});

            result.scores = final_scores.index({torch::indexing::Slice(None, 100)});

            result.classes = result.classes.index(idx);
            result.boxes = result.boxes.index(idx);
            result.masks = result.masks.index(idx);
        }

        InstanceSegmentationResult Yolact::detect(int batch_idx, torch::Tensor confPreds, torch::Tensor decoded_boxes, torch::Tensor maskTensor)
        {

            InstanceSegmentationResult result;

            auto cur_scores = confPreds[batch_idx];

            cur_scores = cur_scores.index({{torch::indexing::Slice(1, None),
                                            torch::indexing::Slice(None)}});

            //this is a tuple
            auto [conf_scores, conf_index] = torch::max(cur_scores, 0);

            torch::Tensor keep = {conf_scores > 0.4};

            result.scores = cur_scores.index({torch::indexing::Slice(None),
                                              keep});

            result.boxes = decoded_boxes.index({keep, torch::indexing::Slice(None)});
            //[0] is the batch size element. If one element is 0
            result.masks = maskTensor[0].index({keep, torch::indexing::Slice(None)});

            if (!result.scores.numel())
            {

                InstanceSegmentationResult tensorNumel = {};
                return tensorNumel;

                if (result.scores.sizes()[1] == 0)
                {

                    InstanceSegmentationResult tensor = {};
                    return tensor;
                }
            }

            FastNms(result, m_fNmsThresh);

            return result;
        }
        InstanceSegmentationResult Yolact::postprocessing(string imagePathAccuracy)
        {
            torch::Tensor locTensor = torch::from_blob((float *)(m_fpOutOnnxRuntime[0]), {1, 19248, 4}).clone();
            torch::Tensor confTensor = torch::from_blob((float *)(m_fpOutOnnxRuntime[1]), {1, 19248, m_iNumClasses + 1}).clone();
            torch::Tensor maskTensor = torch::from_blob((float *)(m_fpOutOnnxRuntime[2]), {1, 19248, 32}).clone();
            torch::Tensor priorsTensor = torch::from_blob((float *)(m_fpOutOnnxRuntime[3]), {19248, 4}).clone();
            torch::Tensor protoTensor = torch::from_blob((float *)(m_fpOutOnnxRuntime[4]), {1, 138, 138, 32}).clone();

            int batch_size = locTensor.sizes()[0];
            int num_priors = priorsTensor.sizes()[0];

            //cout << "NUM CLASSES " << num_priors << " " << batch_size << endl;

            auto confPreds = confTensor.view({batch_size, num_priors, m_iNumClasses + 1}).transpose(2, 1).contiguous();

            //decode Function

            torch::Tensor decoded_boxes = decode(locTensor, priorsTensor, 0);

            //fast_nms------------------------------

            int topk = 200;

            auto result = detect(0, confPreds, decoded_boxes, maskTensor);

            if (!result.scores.numel())
            {

                InstanceSegmentationResult tensor = {};
                return tensor;
            }

            //all score above threshold
            auto final_keep = (result.scores > m_fDetectionThresh);

            auto final_classes = result.classes.index(final_keep);
            auto final_boxes = result.boxes.index(final_keep);
            auto final_mask_nms = result.masks.index(final_keep);

            result.classes = result.classes.index(final_keep);
            result.boxes = result.boxes.index(final_keep);
            result.masks = result.masks.index(final_keep);

            result.proto = protoTensor[0];

            //Find Accuracy for COCO

#ifdef EVAL_ACCURACY
            //need for handling image path for COCO DATASET
            //for every image
            //cout << "eval accuracy" << endl;
            string image_id = imagePathAccuracy;

            auto scores = result.scores.index(final_keep);

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

            auto resultBbox = getCorrectBbox(result);

            for (int i = 0; i < result.boxes.sizes()[0]; i++)
            {

                Json::Value root;
                // Json::Value categoryIdJson;
                // Json::Value bboxJson;
                // Json::Value score;
                root["image_id"] = std::stoi(image_id);

                int cocoCategory = 0;
                //darknet has 80 class while coco has 90 classes. We need to handle different number of classes on output
                //1

                cocoCategory = CocoMap[result.classes[i].item<int>()];

                root["category_id"] = cocoCategory;

                Json::Value valueBBoxjson(Json::arrayValue);

                valueBBoxjson.append(resultBbox[i].x);
                valueBBoxjson.append(resultBbox[i].y);
                valueBBoxjson.append(resultBbox[i].width);
                valueBBoxjson.append(resultBbox[i].height);

                root["bbox"] = valueBBoxjson;
                root["score"] = scores[i].item<float>();

                m_JsonRootArray.append(root);
            }

#endif

            return result;
        }

        void Yolact::sanitizeCoordinate(torch::Tensor &x, torch::Tensor &y, int imageDimension)
        {

            auto x1 = x * imageDimension;
            auto y1 = y * imageDimension;

            auto bbox_x = torch::min(x1, y1);
            auto bbox_y = torch::max(x1, y1);

            x = torch::clamp(bbox_x, 0);
            y = torch::clamp(bbox_y, 0, imageDimension);
        }

        vector<Rect> Yolact::getCorrectBbox(InstanceSegmentationResult result)
        {
            vector<Rect> resultCvBbox;

            auto x = result.boxes.index({torch::indexing::Slice(None), 0});
            auto y = result.boxes.index({torch::indexing::Slice(None), 1});
            auto width = result.boxes.index({torch::indexing::Slice(None), 2});
            auto height = result.boxes.index({torch::indexing::Slice(None), 3});

            sanitizeCoordinate(x, width, m_ImageWidhtOrig);
            sanitizeCoordinate(y, height, m_ImageHeightOrig);

            for (int i = 0; i < x.sizes()[0]; i++)
            {

                int x_rect = x[i].item<int>();
                int y_rect = y[i].item<int>();
                int width_rect = width[i].item<int>() - x[i].item<int>();
                int height_rect = height[i].item<int>() - y[i].item<int>();

                Rect rect(x_rect, y_rect, width_rect, height_rect);

                // if (!image.empty())
                //     rectangle(image, rect, (255, 255, 255), 0.5);

                resultCvBbox.push_back(rect);
            }

            return resultCvBbox;
        }

        void Yolact::cropMask(torch::Tensor &masks, torch::Tensor boxes)
        {

            int h = masks.sizes()[0];
            int w = masks.sizes()[1];
            int n = masks.sizes()[2];

            auto x = boxes.index({torch::indexing::Slice(None), 0});
            auto y = boxes.index({torch::indexing::Slice(None), 1});
            auto width = boxes.index({torch::indexing::Slice(None), 2});
            auto height = boxes.index({torch::indexing::Slice(None), 3});

            sanitizeCoordinate(x, width, w);
            sanitizeCoordinate(y, height, h);

            auto rows = torch::arange(w).view({1, -1, 1}).expand({h, w, n});
            auto cols = torch::arange(h).view({-1, 1, 1}).expand({h, w, n});

            auto mask_left = rows >= x.view({1, 1, -1});
            auto mask_right = rows < width.view({1, 1, -1});
            auto mask_up = cols >= y.view({1, 1, -1});
            auto mask_down = cols < height.view({1, 1, -1});

            auto crop_mask = mask_left * mask_right * mask_up * mask_down;

            //cout << "CROP MASK SIZE" << crop_mask.sizes() << endl;

            masks = masks * crop_mask.to(torch::kFloat);
        }

        /*
        
        return a vector<Mat> resultMask where each value is a Mat CV_8UC1
        
        */
        vector<Mat> Yolact::getCorrectMask(InstanceSegmentationResult result)
        {

            vector<Mat> resultcvMask;

            auto masks = torch::matmul(result.proto, result.masks.t());

            masks = torch::sigmoid(masks);

            cropMask(masks, result.boxes);

            masks = masks.permute({2, 0, 1}).contiguous();

            masks = torch::nn::functional::interpolate(masks.unsqueeze(0),
                                                       torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{m_ImageHeightOrig, m_ImageWidhtOrig}).mode(torch::kBilinear).align_corners(false))
                        .squeeze(0);

            masks = masks.gt(0.5);

            for (int i = 0; i < masks.sizes()[0]; i++)
            {

                //masks=masks.unsqueeze(0);
                auto tensor = masks[i].mul(255).to(torch::kU8);
                //Devo convertire il tensore in Cpu se voglio visualizzarlo con OpenCv
                tensor = tensor.to(torch::kCPU);
                cv::Mat resultImg(m_ImageHeightOrig, m_ImageWidhtOrig, CV_8UC1);
                std::memcpy((void *)resultImg.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());

                resultcvMask.push_back(resultImg);
            }

            return resultcvMask;
        }

        void Yolact::createAccuracyFile()
        {

            Json::StreamWriterBuilder builder;
            const std::string json_file = Json::writeString(builder, m_JsonRootArray);

            ofstream myfile;
            myfile.open("yolactVal.json", std::ios::in | std::ios::out | std::ios::app);
            myfile << json_file + "\n";
            myfile.close();
        }

        Yolact::~Yolact()
        {

            m_OrtSession.reset();
            m_OrtEnv.reset();
        }

    } // namespace instanceSegmentation

} // namespace ai4prod