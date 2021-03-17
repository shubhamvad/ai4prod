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
using namespace std::chrono;

namespace ai4prod
{

    namespace instanceSegmentation
    {

        class AIPRODUCTION_EXPORT Yolact : ai4prod::modelInterfaceInstanceSegmentation
        {

        private:
            
            

            ai4prod::aiutils m_aut;
            MODE m_eMode;
            string m_sMessage;

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
            int m_iMrows;
            int m_iMcols;

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
            float *m_fpOutOnnxRuntime[5];

            size_t m_InputTorchTensorSize;

            //LIBTORCH Tensor

            torch::Tensor m_TInputTensor;
            //torch::Tensor m_TOutputTensor;

            //imageDimension Original
            int m_ImageWidhtOrig;
            int m_ImageHeightOrig;

            //Test VARIABLE TO BE ELIMNATED

        

            //FUNCTION
            void setOnnxRuntimeEnv();
            void createYamlConfig(std::string modelPathOnnx, int input_h, int input_w, int numClasses,MODE t, std::string model_path);
            //void setEnvVariable();
            void setSession();
            void setOnnxRuntimeModelInputOutput();

            //Yolact Result custom Struct

            //postprocessing

            torch::Tensor decode(torch::Tensor &locTensor, torch::Tensor &priorsTensor, int batchSizePos=0);

            InstanceSegmentationResult detect(int batch_idx, torch::Tensor confPreds, torch::Tensor decoded_boxes, torch::Tensor maskTensor);

            void FastNms(InstanceSegmentationResult &result, float nms_thres, int topk = 200);

            torch::Tensor jaccard(torch::Tensor boxes_a, torch::Tensor boxes_b);

            torch::Tensor intersect(torch::Tensor box_a, torch::Tensor box_b);

            //display

            void sanitizeCoordinate(torch::Tensor &x,torch::Tensor &y, int imageDimension);
            void cropMask(torch::Tensor &masks,torch::Tensor boxes);
            
            //detection accuracy

             //array with all detection accuracy
            Json::Value m_JsonRootArray;

            

        public:
            Yolact();
            virtual ~Yolact();
            bool init(std::string modelPathOnnx, int input_h, int input_w,int numClasses, MODE t, std::string model_path = NULL);

            void preprocessing(Mat &Image);
            void runmodel();
            InstanceSegmentationResult postprocessing(string imagePathAccuracy="");

            //display 

            vector<Rect> getCorrectBbox(InstanceSegmentationResult result);

            vector<Mat> getCorrectMask(InstanceSegmentationResult result);

            //detection accuracy
            
            string m_sAccurayImagePath;
            void createAccuracyFile();
        };

    } // namespace instanceSegmentation

} // namespace ai4prod
