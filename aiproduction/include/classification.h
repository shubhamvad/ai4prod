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

namespace ai4prod
{

   namespace classification
   {

      class AIPRODUCTION_EXPORT ResNet50 : ai4prod::modelInterfaceClassification
      {

      private:

         //INIT VARIABLE

         int m_iNumClasses;
         int m_iNumberOfReturnedPrediction;

         YAML::Node m_ymlConfig;
         std::string m_sModelTrPath;
         std::string m_sModelOnnxPath;
         std::string m_sEngineFp;
         std::string m_sEngineCache;

         MODE m_eMode;
         
         int m_iInput_h;
         int m_iInput_w;
         int m_iCropImage;


         //ONNX RUNTIME

         Ort::SessionOptions m_OrtSessionOptions;
         Ort::AllocatorWithDefaultOptions allocator;
         
         std::unique_ptr<Ort::Session> m_OrtSession;
         std::unique_ptr<Ort::Env> m_OrtEnv;

         
         //OnnxRuntimeModelloInput

         size_t num_input_nodes;
         std::vector<const char *> input_node_names;
         std::vector<int64_t> input_node_dims;

         //OnnxRuntimeModelloOutput

         size_t num_out_nodes;
         std::vector<const char *> out_node_names;
         std::vector<int64_t> out_node_dims;

         //onnxruntime data
         float *m_fpOutOnnxRuntime;
         float *m_fpInOnnxRuntime;

         //Dimensione del tensore di input modello .onnx
         size_t input_tensor_size;


         //LIBTORCH DATA

         torch::Tensor inputTensor;
         torch::Tensor m_TOutputTensor;


         //THREAD SAFE

         //handle initialization
         bool m_bInit;
         //used to call init only one time per instances
         bool m_bCheckInit;
         //used to verify if preprocess is called on the same run
         bool m_bCheckPre;
         //used to verify if run model is called on the same run
         bool m_bCheckRun;
         //used to verify id post process is called
         bool m_bCheckPost;

         //Utils
         ai4prod::aiutils aut;

      
         //ERROR HANDLING

         // message
         std::string m_sMessage;


         // FUNCTION


         //init Function

         void setOnnxRuntimeEnv();
         void setOnnxRuntimeModelInputOutput();
         bool checkParameterConfig(std::string modelPathOnnx, int input_w, int input_h, int numClasses, MODE t);
         bool createYamlConfig(std::string modelPathOnnx, int input_w, int input_h, int numClasses, int NumberOfReturnedPrediction, MODE t, std::string modelTr_path);
         void setEnvVariable();
         void setSession();



      public:
         ResNet50();

         virtual ~ResNet50();

         bool init(std::string modelPath, int width, int height, int numClasses, int NumberOfReturnedPrediction, MODE t, std::string modelTr_path = NULL);

         std::string m_sAccurayImagePath;

         void preprocessing(cv::Mat &Image);
         std::tuple<torch::Tensor, torch::Tensor> postprocessing();
         void runmodel();

         std::string getMessage(){

            return m_sMessage;

         }
      };

   } //namespace classification

} // namespace ai4prod