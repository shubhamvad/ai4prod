#include "classification.h"

#include "../onnxruntime/include/onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"
#include "../onnxruntime/include/onnxruntime/core/providers/providers.h"

using namespace std;

using namespace onnxruntime;

namespace classification
{

    //Da fare

    /*
1) Costruttore inserire la path di caricamento del modello
2) Sistemare il codice per caricare il modello Resnet
*/

    ResNet50::ResNet50()
    {

        //inizializzazione sessione OnnxRuntime
        //const char * model_path= "/home/tondelli/Desktop/2020/aiproductionready/onnxruntime/model/cpu/squeezenet.onnx";

        //Ort::Env env(ORT_LOGGING_LEVEL_FATAL, "test");
        //session = new Ort::Session(env, model_path, session_options);

        //session_options.SetIntraOpNumThreads(6);
        //session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }

    ResNet50::ResNet50(std::string path, std::string modelTr_path)
    {

        //queste variabili devono essere settate prima di inzializzara la sessione
        char cacheModel[] = "ORT_TENSORRT_ENGINE_CACHE_ENABLE=1";
        putenv(cacheModel);

        m_sModelTrPath = "ORT_TENSORRT_ENGINE_CACHE_PATH="+ modelTr_path;

        cout<<m_sModelTrPath<<endl;

        int n = m_sModelTrPath.length();
        char modelSavePath[n + 1];

        strcpy(modelSavePath, m_sModelTrPath.c_str());
        //esporto le path del modello di Tensorrt
        putenv(modelSavePath);

        
        //test = OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options,0);

        env = new Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "test");

        //le opzioni devono essere settate prima della creazione della sessione

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

        session = new Ort::Session(*env, path.c_str(), session_options);

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

    /*
Distruttore
*/
    ResNet50::~ResNet50()
    {
    }

    torch::Tensor ResNet50::preprocessing(Mat &Image)
    {

        //ResNet50::model=data;

        inputTensor = convertMatToTensor(Image, Image.cols, Image.rows, Image.channels(), 1);

        //definisco la dimensione di input

        input_tensor_size = Image.cols * Image.rows * Image.channels();

        //Mat testImage;

        //testImage = convertTensortToMat(inputTensor, 224, 224);

        //imshow("test image", testImage);
        //imshow("original", Image);
        //waitKey(0);

        //verifico che immagine sia la stessa
        //equalImage(Image, testImage);

        //se le 2 immagini sono uguali allora noramlizzo il tensore
        //questi sono i valori di ImageNet

        inputTensor[0][0] = inputTensor[0][0].sub_(0.485).div_(0.229);
        inputTensor[0][1] = inputTensor[0][1].sub_(0.456).div_(0.224);
        inputTensor[0][2] = inputTensor[0][2].sub_(0.406).div_(0.225);

        //cout << "preprocessing" << endl;

        //cout<<session->GetInputCount()<<endl;

        //cout<<session.GetInputCount()<<endl;

        //Preprocessig Image

        return inputTensor;
    }

    torch::Tensor ResNet50::runmodel(torch::Tensor &input)
    {

        //verifico che il tensore sia contiguous()

        if (input.is_contiguous())
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
                printf("Input %d : name=%s\n", i, input_name);

                Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                ONNXTensorElementDataType type = tensor_info.GetElementType();
                printf("Input %d : type=%d\n", i, type);

                // print input shapes/dims
                out_node_dims = tensor_info.GetShape();
                printf("Input %d : num_dims=%zu\n", i, out_node_dims.size());
                for (int j = 0; j < out_node_dims.size(); j++)
                    printf("Input %d : dim %d=%jd\n", i, j, out_node_dims[j]);
            }

            std::vector<const char *> output_node_names = {"output1"};

            // initialize input data with values in [0.0, 1.0]
            //for (unsigned int i = 0; i < input_tensor_size; i++)
            //    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

            // create input tensor object from data values
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
            assert(input_tensor.IsTensor());

#ifdef TENSORRT

            //conversione string to char[]
            // m_sModelTrPath= "ORT_TENSORRT_ENGINE_CACHE_PATH='" + m_sModelTrPath + "'";
            // int n = m_sModelTrPath.length();
            // char modelSavePath[n + 1];

            // strcpy(modelSavePath,m_sModelTrPath.c_str());
            // //esporto le path del modello di Tensorrt
            // putenv(modelSavePath );

            // std::cout << "TEMP = " << getenv("ORT_TENSORRT_ENGINE_CACHE_PATH") << std::endl;

#endif

            clock_t start, end;

            start = clock();
            auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
            end = clock();

            assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

            double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
            cout << "Time taken by program is : " << fixed
                 << time_taken << setprecision(5);
            cout << " sec " << endl;

            assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

            float *floatarr = output_tensors.front().GetTensorMutableData<float>();
            float label;

            int cls;

            // score the model, and print scores for first 5 classes
            for (int i = 0; i < 1000; i++)

                if (floatarr[i] > label)
                {

                    printf("Score for class [%d] =  %f\n", i, floatarr[i]);

                    label = floatarr[i];
                    cls = i;
                }

            // Results should be as below...
            // Score for class[0] = 0.000045
            // Score for class[1] = 0.003846
            // Score for class[2] = 0.000125
            // Score for class[3] = 0.001180
            // Score for class[4] = 0.001317
            printf("Done!\n");
            cout << "CLASSEINSIDE: " << cls << endl;
            // //Conversione del tensore

            // p = static_cast<float *>(inputTensor.storage().data());

            // std::vector<float> input_tensor_values(input_onnx_size);

            //  for (int i = 0; i < input_onnx_size; i++) {

            //       input_tensor_values[i] = p[i];

            //       cout<<p[i]<<endl;

            // }

            // //INPUT
            // size_t num_input_nodes = session->GetInputCount();
            // std::vector<const char *> input_node_names(num_input_nodes);
            // std::vector<int64_t> input_node_dims;

            // //OUTPUT

            // size_t num_out_nodes = session->GetOutputCount();
            // std::vector<const char *> out_node_names(num_out_nodes);
            // std::vector<int64_t> out_node_dims;

            // std::vector<const char *> output_node_names = {"output1"};

            // auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_onnx_size, input_node_dims.data(), 4);

            // vector<Ort::Value> out;

            // out = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);

            // out.size();
            // out.front().IsTensor();

            // float *floatarr = out.front().GetTensorMutableData<float>();

            // torch::Tensor prova;

            // //con .clone() i valori sono clonati nel tensore
            // prova = torch::from_blob(floatarr, {1000}).clone();
        }

        else
        {

            cout << "Il tensore non è contiguous non possibile eseguire inferenza" << endl;
        }

        torch::Tensor test = torch::rand({2, 3});

        return test;
    }

    torch::Tensor ResNet50::postprocessing(torch::Tensor &output)
    {

        torch::Tensor test = torch::rand({2, 3});
        return test;
    }

} // namespace classification