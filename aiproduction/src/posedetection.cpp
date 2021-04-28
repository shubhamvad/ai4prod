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

#include "posedetection.h"

#ifdef TENSORRT
#include "../../deps/onnxruntime/tensorrt/include/onnxruntime/core/providers/providers.h"
#include "../../deps/onnxruntime/tensorrt/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"

#endif

#ifdef DIRECTML
#include "../../deps/onnxruntime/directml/include/onnxruntime/core/providers/providers.h"
#include "../../deps/onnxruntime/directml/include/onnxruntime/core/providers/dml/dml_provider_factory.h"

#endif

namespace ai4prod
{
    namespace poseDetection
    {
        Hrnet::Hrnet()
        {
        }
        bool Hrnet::checkParameterConfig(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path)
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

        //create config file if not present
        //return false if something is not configured correctly
        bool Hrnet::createYamlConfig(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path)
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

        void Hrnet::setOnnxRuntimeEnv()
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

        void Hrnet::setSession()
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

        void Hrnet::setOnnxRuntimeModelInputOutput()
        {
            m_num_input_nodes = m_OrtSession->GetInputCount();
            m_input_node_names = std::vector<const char *>(m_num_input_nodes);

            m_num_out_nodes = m_OrtSession->GetOutputCount();

            m_output_node_names = std::vector<const char *>(m_num_out_nodes);
        }

        bool Hrnet::init(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path)
        {
            if (!m_aut.createFolderIfNotExist(model_path))
            {

                std::cout << "cannot create folder" << std::endl;

                return false;
            }

            std::cout << "INIT MODE " << t << std::endl;

            if (!createYamlConfig(modelPathOnnx, input_h, input_w, numClasses, t, model_path))
            {

                std::cout << m_sMessage << std::endl;
                return false;
            }

            //verify if Mode is implemented
            if (!m_aut.checkMode(m_eMode, m_sMessage))
            {

                std::cout << m_sMessage << std::endl;
                return false;
            }

#ifdef __linux__

            std::string cacheModel = "ORT_TENSORRT_ENGINE_CACHE_ENABLE=" + m_sEngineCache;

            int cacheLenght = cacheModel.length();
            char cacheModelchar[cacheLenght + 1];
            strcpy(cacheModelchar, cacheModel.c_str());
            putenv(cacheModelchar);

            std::string fp16 = "ORT_TENSORRT_FP16_ENABLE=" + m_sEngineFp;
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
            std::cout << "1" << std::endl;
            setOnnxRuntimeEnv();
            std::cout << "2" << std::endl;
            setSession();
            std::cout << "3" << std::endl;
            setOnnxRuntimeModelInputOutput();
            std::cout << "initDone" << std::endl;
            return true;
        }

        void Hrnet::boxToCenterScale(torch::Tensor &result, std::vector<cv::Point2f> &centers, std::vector<cv::Point2f> &scales)
        {

            for (int i = 0; i < result.sizes()[0]; i++)
            {

                float x = result[i][0].item<float>();
                float y = result[i][1].item<float>();
                float box_width = result[i][2].item<float>();
                float box_height = result[i][3].item<float>();

                float center_x = x + box_width * 0.5;
                float center_y = y + box_height * 0.5;

                //parameter of the network
                float pixel_std = 200.0;

                float aspect_ratio = (float)m_iInput_w / (float)m_iInput_h;

                std::cout <<"Aspect ratio "<< aspect_ratio<<std::endl;

                if (box_width > aspect_ratio * box_height)
                {
                    std::cout <<"width "<< aspect_ratio<<std::endl;
                    box_height = box_width / aspect_ratio;
                }
                else
                {   
                    std::cout <<"height "<< aspect_ratio<<std::endl;
                    box_width = box_height * aspect_ratio;
                }

                if (center_x != -1)
                    scales.push_back(cv::Point2f((box_width / pixel_std), (box_height / pixel_std)) * 1.25);

                centers.push_back(cv::Point2f(center_x, center_y));
            }
        }

        cv::Point2f Hrnet::getDir(cv::Point2f srcPoint,float rot){

            double sn= sin(rot);
            double cs= cos(rot);

            cv::Point2f resultPoint;

            resultPoint.x= srcPoint.x*cs - srcPoint.y*sn;
            resultPoint.y= srcPoint.x*sn + srcPoint.y*cs;

            return resultPoint;

        }

        cv::Point2f Hrnet::get3rdPoint(cv::Point2f first, cv::Point2f second){

            cv::Point2f direct= first -second;

            return second + cv::Point2f(-direct.y,direct.x);
            
        }
        /*
        getAffineTransform to process input bbox respect to model input
         give center and scale of single bbox return AffineTransform
        */
        cv::Mat Hrnet::getAffineTransform(cv::Point2f center,cv::Point2f scale){

            float rot=0;
            float shift= 0;
            cv::Point2f tmpScale = scale*200;

            float src_w= tmpScale.x;

            float dst_w= (float)m_iInput_w;
            float dst_h= (float)m_iInput_h;
            

            float rot_rad= 3.141592 * rot/180;

            cv::Point2f src_dir= getDir(cv::Point2f(0,src_w*(-0.5)),rot_rad);

            cv::Point2f dst_dir= cv::Point2f(0,dst_w*(-0.5));

            cv::Point2f srcTri[3];

            srcTri[0]= center + tmpScale*shift;
            srcTri[1]= center + src_dir +tmpScale*shift;
            srcTri[2]= get3rdPoint( srcTri[0],srcTri[1]);

            cv::Point2f dstTri[3];

            dstTri[0]= cv::Point2f(dst_w*0.5,dst_h*0.5);
            dstTri[1]= cv::Point2f(dst_w*0.5,dst_h*0.5) +dst_dir;
            dstTri[2]= get3rdPoint( dstTri[0],dstTri[1]);

            for(int i=0; i < 3;i++){

                std::cout<< "SRC "<< srcTri[i].x<< " " << srcTri[i].y<<std::endl;
                std::cout<< "DST "<< dstTri[i].x<< " " << dstTri[i].y<<std::endl;
            }

            return cv::getAffineTransform(srcTri,dstTri);


        }


        /*
            Image: Image to be processed
            result:  list of all people detected bbox usually from an object detector. Coordinate must be respect of the original image size
        */
        void Hrnet::preprocessing(cv::Mat &Image, torch::Tensor result)
        {
            

            m_iInputOrig_h=Image.cols;
            m_iInputOrig_w=Image.rows;

            std::vector<cv::Point2f> centers;
            std::vector<cv::Point2f> scales;

            //computer bbox center and scales

            boxToCenterScale(result,centers,scales);



            std::vector<cv::Mat> bboxWarp;
            for (int i=0; i< centers.size();i++){
                
                std::cout <<"CENTER "<< centers[i].x << " " <<centers[i].y<<std::endl;
                std::cout <<"SCALES " <<scales[i].x << " " <<scales[i].y<<std::endl;

                cv::Mat trans= getAffineTransform(centers[i],scales[i]);

                std::cout << "TRANS "<< trans<<std::endl;
                cv::Mat tmp= cv::Mat::zeros( m_iInput_w, m_iInput_w, Image.type() );
                
                cv::warpAffine(Image,tmp,trans,cv::Size(m_iInput_w,m_iInput_h));

                bboxWarp.push_back(tmp);

               
            }

            std::cout << bboxWarp.size()<<std::endl;
            for (int i=0;i< bboxWarp.size();i++){

                 cv::imshow("bboxWarped",bboxWarp[i]);
                cv::waitKey(0);
            }

            //convert each bbox to a tensor
            
            for (int i=0; i< bboxWarp.size();i++){

                torch::Tensor tmp=m_aut.convertMatToTensor(bboxWarp[i],bboxWarp[i].cols,bboxWarp[i].rows,bboxWarp[i].channels(),1);
                

                
                //Normalize
                tmp[0][0] = tmp[0][0].sub_(0.485).div_(0.229);
                tmp[0][1] = tmp[0][1].sub_(0.456).div_(0.224);
                tmp[0][2] = tmp[0][2].sub_(0.406).div_(0.225);


                m_TInputTensor.push_back(tmp);
            }


        }

        void Hrnet::runmodel()
        {


        }

        torch::Tensor Hrnet::postprocessing(std::string imagePathAccuracy)
        {

            torch::Tensor t;
            return t;
        }

        Hrnet::~Hrnet()
        {
            m_OrtSession.reset();
            m_OrtEnv.reset();
        }
    } //poseEstimation

} //Ai4prod