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
using namespace torch::indexing;

namespace ai4prod
{

    namespace instanceSegmentation
    {

        class AIPRODUCTION_EXPORT Yolact : ai4prod::modelInterfaceInstanceSegmentation
        {

        private:

            struct TensorResult{

                torch::Tensor boxes;
                torch::Tensor masks;
                torch::Tensor classes;
                torch::Tensor scores;
            };

            TensorResult m_TensorResult;

            ai4prod::aiutils m_aut;
            MODE m_eMode;
            string m_sMessage;

            YAML::Node m_ymlConfig;
            std::string m_sModelTrPath;
            std::string m_sModelOnnxPath;
            std::string m_sEngineFp;
            std::string m_sEngineCache;

            //neural network input dimension

            int m_iInput_h;
            int m_iInput_w;
            //original image width and height
            int m_iMrows;
            int m_iMcols;

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
            float *m_fpOutOnnxRuntime[5];

            size_t m_InputTorchTensorSize;

            //LIBTORCH Tensor

            torch::Tensor m_TInputTensor;
            //torch::Tensor m_TOutputTensor;

            //imageDimension Original
            int m_ImageWidhtOrig;
            int m_ImageHeightOrig;

            //Test VARIABLE TO BE ELIMNATED

            Mat testImage;

            //FUNCTION
            void setOnnxRuntimeEnv();
            void createYamlConfig(std::string modelPathOnnx, int input_h, int input_w, MODE t, std::string model_path);
            //void setEnvVariable();
            void setSession();
            void setOnnxRuntimeModelInputOutput();

            
            //postprocessing

            torch::Tensor decode(torch::Tensor locTensor, torch::Tensor priorsTensor);
            
            TensorResult detect(int batch_idx,torch::Tensor confPreds, torch::Tensor decoded_boxes, torch::Tensor maskTensor);

            void FastNms(TensorResult &result,float nms_thres,int topk=200);

            torch::Tensor jaccard(torch::Tensor boxes_a, torch::Tensor boxes_b);

            torch::Tensor intersect(torch::Tensor box_a,torch::Tensor box_b);

        public:
            Yolact();
            virtual ~Yolact();
            bool init(std::string modelPathOnnx, int input_h, int input_w, MODE t, std::string model_path = NULL);

            void preprocessing(Mat &Image);
            torch::Tensor postprocessing();
            void runmodel();
        };

    } // namespace instanceSegmentation

} // namespace ai4prod
