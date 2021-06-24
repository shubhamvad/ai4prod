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

#include "semanticsegmentation.h"
#include "defines.h"

namespace ai4prod
{
    namespace semanticSegmentation
    {
        USquaredNet::USquaredNet()
        {
        }

        //verify if all parameters are good.
        //for example if Mode: tensorRt is the same as initialization
        bool USquaredNet::checkParameterConfig(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path)
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
        bool USquaredNet::createYamlConfig(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path)
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
        void USquaredNet::setOnnxRuntimeEnv()
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

        void USquaredNet::setSession()
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
        void USquaredNet::setOnnxRuntimeModelInputOutput()
        {
            m_num_input_nodes = m_OrtSession->GetInputCount();
            m_input_node_names = std::vector<const char *>(m_num_input_nodes);

            m_num_out_nodes = m_OrtSession->GetOutputCount();

            m_output_node_names = std::vector<const char *>(m_num_out_nodes);
        }
        bool USquaredNet::init(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path)
        {

            if (!m_aut.createFolderIfNotExist(model_path))
            {

                std::cout << "cannot create folder" << std::endl;

                return false;
            }

            std::cout << "INIT MODE " << t << std::endl;
            //create config file and check for configuration error
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

            setOnnxRuntimeEnv();

            setSession();

            setOnnxRuntimeModelInputOutput();

            std::cout << "initDone" << std::endl;
            return true;
        }

        void USquaredNet::preprocessing(cv::Mat &Image)
        {

            //guassian blur

            float truncate = 4.0;

            std::vector<float> inputShape = {(float)Image.rows, (float)Image.cols, (float)Image.channels()};
            std::vector<float> outputShape = {(float)m_iInput_h, (float)m_iInput_w, 3.0};

            std::vector<float> factors = {inputShape[0] / outputShape[0], inputShape[1] / outputShape[1], inputShape[2] / outputShape[2]};

            std::vector<float> sigma;

            for (int i = 0; i < factors.size(); i++)
            {

                sigma.push_back(std::max((float)0.0, (factors[i] - 1) / 2));
            }

            std::cout << sigma << std::endl;

            int sizeB = (int)truncate * sigma[2];
            int sizeG = (int)truncate * sigma[1];
            int sizeR = (int)truncate * sigma[0];

            if (sizeB == 0 || sizeB % 2 == 0)
            {
                sizeB = sizeB + 1;
            }

            if (sizeG == 0 || sizeG % 2 == 0)
            {
                sizeG = sizeG + 1;
            }

            if (sizeR == 0 || sizeR % 2 == 0)
            {
                sizeR = sizeR + 1;
            }


            cv::Mat rgbchannel[3];
            // The actual splitting.
            cv::split(Image, rgbchannel);

            cv::GaussianBlur(rgbchannel[0], rgbchannel[0],cv::Size(sizeB,1),sigma[2],sigma[2],0);
            cv::GaussianBlur(rgbchannel[1], rgbchannel[1],cv::Size(sizeG,1),sigma[1],sigma[1],0);
            cv::GaussianBlur(rgbchannel[2], rgbchannel[2],cv::Size(sizeR,1),sigma[0],sigma[0],0);
            //gaussian filter

            cv::Mat merged;
            
            cv::merge(rgbchannel,3,merged);

            //cv::cvtColor(merged,merged,cv::COLOR_RGB2BGR);

            cv::resize(merged,merged,cv::Size(m_iInput_h,m_iInput_w));

            cv::imwrite("imageCplus.jpg",merged);
           
        }

        void USquaredNet::runmodel()
        {
        }

        torch::Tensor USquaredNet::postprocessing(std::string imagePathAccuracy)
        {

            torch::Tensor t;

            return t;
        }

        USquaredNet::~USquaredNet()
        {
            m_OrtSession.reset();
            m_OrtEnv.reset();
        }

    }
}