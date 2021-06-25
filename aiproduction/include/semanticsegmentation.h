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

#include "modelInterface.h"

using namespace std::chrono;

namespace ai4prod
{

    namespace semanticSegmentation
    {
        class AIPRODUCTION_EXPORT USquaredNet : ai4prod::modelInterfaceSemanticSegmentation
        {

        private:
            ai4prod::aiutils m_aut;
            MODE m_eMode;
            std::string m_sMessage;

            //Config Parameter
            YAML::Node m_ymlConfig;
            std::string m_sModelTrPath;
            std::string m_sModelOnnxPath;
            std::string m_sEngineFp;
            std::string m_sEngineCache;

            //neural network input dimension

            int m_iInput_h;
            int m_iInput_w;
            //original image width and height
            int m_iOrig_h;
            int m_iOrig_w;

            int m_iNumClasses;
            float m_fNmsThresh;
            float m_fDetectionThresh;

            //ONNXRUNTIME
            //onnxRuntime Session

            Ort::SessionOptions m_OrtSessionOptions;
            Ort::AllocatorWithDefaultOptions allocator;

            std::unique_ptr<Ort::Session> m_OrtSession;
            std::unique_ptr<Ort::Env> m_OrtEnv;

            //OnnxRuntime Input Model

            size_t m_num_input_nodes;
            std::vector<const char *> m_input_node_names;

            //OnnxRuntime Output Model

            size_t m_num_out_nodes;
            std::vector<const char *> m_output_node_names;

            //session IN/OUT

            float *m_fpInputOnnxRuntime;
            float *m_fpOutOnnxRuntime[7];

            size_t m_InputTorchTensorSize;

            //LIBTORCH Tensor

            torch::Tensor m_TInputTensor;
            //torch::Tensor m_TOutputTensor;

            //imageDimension Original
            int m_ImageWidhtOrig;
            int m_ImageHeightOrig;

            //------------------METHOD--------------------------
            void setOnnxRuntimeEnv();
            bool checkParameterConfig(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path);
            bool createYamlConfig(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path);
            //void setEnvVariable();
            void setSession();
            void setOnnxRuntimeModelInputOutput();


           


        public:
            USquaredNet();
            virtual ~USquaredNet();
            bool init(std::string modelPathOnnx, int input_h, int input_w, int numClasses, MODE t, std::string model_path = NULL);

            void preprocessing(cv::Mat &Image);
            void runmodel();
            torch::Tensor postprocessing(std::string imagePathAccuracy = "");

             //Post processing
            cv::Mat convertPredToMask(torch::Tensor &result);
        };

    }//U2Squared
}//Ai4prod