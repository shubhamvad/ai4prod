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
        }

        void Yolact::preprocessing(Mat &Image)
        {
            Mat tmpImage;
            tmpImage = Image.clone();

            //tensor with RGB channel
            m_TInputTensor = m_aut.convertMatToTensor8bit(tmpImage, tmpImage.cols, tmpImage.rows, tmpImage.channels(), 1);
            //torch::nn::Functional::InterpolateFuncOptions().scale_factor({ 500 }).mode(torch::kBilinear).align_corners(false);

            m_TInputTensor = torch::nn::functional::interpolate(m_TInputTensor, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{500, 500}).mode(torch::kBilinear).align_corners(false));

            //normalization pixel image are in range [0,255]

            m_TInputTensor[0][0] = m_TInputTensor[0][0].sub_(123.8).div_(58.40);
            m_TInputTensor[0][1] = m_TInputTensor[0][1].sub_(116.78).div_(57.12);
            m_TInputTensor[0][2] = m_TInputTensor[0][2].sub_(103.94).div_(57.38);
        }
        torch::Tensor Yolact::postprocessing()
        {
            auto tensor = torch::ones({0});
            return tensor;
        }
        void Yolact::runmodel()
        {

            //verify if tensor is contiguous
            if (m_TInputTensor.is_contiguous())
            {
            }
            else
            {

                cout << "error" << endl;
            }
        }

        Yolact::~Yolact()
        {

            m_OrtSession.reset();
            m_OrtEnv.reset();
        }

    } // namespace instanceSegmentation

} // namespace ai4prod