#include "utils.h"

using namespace cv;

/*
Cose da Fare

1) Creare una funzione per verificare se 2 immagini sono uguali o capire quanto sono diverse
per il momento dice solo se le 2 immagini sono uguali

2) Funzione per convertire torch::Tensor come input per onnxruntime

3) Compilare TorchVision e aggiungerlo alle librerie. Verificare le trasformazioni con torchVision


*/

//LIBTORCH

/*
Convert a cv::Mat to torch::Tensor 
image: cvMat in BGR format
width: image_width
height: image_height
channel: channel image
batch: batch size for inference

return: torch::Tensor with the right order of input dimension(B,C,W,H)

1) Passare la Mat con & o senza?
2) https://discuss.pytorch.org/t/libtorch-c-convert-a-tensor-to-cv-mat-single-channel/47701/7
*/

namespace aiProductionReady
{

    torch::Tensor aiutils::convertMatToTensor(Mat ImageBGR, int width, int height, int channel, int batch, bool gpu)
    {

        cv::cvtColor(ImageBGR, ImageBGR, COLOR_BGR2RGB);
        cv::Mat img_float;

        //Conversione dei valori nell'intervallo [0,1] tipico del tensore di Pytorch
        ImageBGR.convertTo(img_float, CV_32F, 1.0 / 255);

        auto img_tensor = torch::from_blob(img_float.data, {batch, height, width, channel}).to(torch::kCPU);

        // you need to be contiguous to have all address memory of tensor sequentially
        img_tensor = img_tensor.permute({0, 3, 1, 2}).contiguous();

        //std::cout << img_tensor.dim() << std::endl;
        //std::cout << img_tensor.sizes() << std::endl;

        return img_tensor;
    }

    /*

return: Image BGR

*/

    cv::Mat aiutils::convertTensortToMat(torch::Tensor tensor, int width, int height)
    {

        //devo controllare che la dimensione di batch sia uguale a 1

        //squeeze(): rimuove tutti i valori con dimensione 1
        tensor = tensor.squeeze().detach().permute({1, 2, 0});
        tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
        //Devo convertire il tensore in Cpu se voglio visualizzarlo con OpenCv
        tensor = tensor.to(torch::kCPU);
        cv::Mat resultImg(height, width, CV_8UC3);
        std::memcpy((void *)resultImg.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());

        return resultImg;
    }

    /*

verify if 2 Mat are equal

*/
    bool aiutils::equalImage(const Mat &a, const Mat &b)
    {
        if ((a.rows != b.rows) || (a.cols != b.cols))
            return false;
        Scalar s = sum(a - b);

        std::cout << s << std::endl;

        return (s[0] == 0) && (s[1] == 0) && (s[2] == 0);
    }

    torch::Tensor aiutils::convert2dVectorToTensor(std::vector<std::vector<float>> &input)
    {

        if (input[0].size() == 0)
        {

            cout << "empty value" << endl;
            auto tensor = torch::empty({200, 200, 3});
            return tensor;
        }

        else
        {

            cv::Mat NewSamples(0, input[0].size(), cv::DataType<float>::type);

            for (unsigned int i = 0; i < input.size(); ++i)
            {
                // Make a temporary cv::Mat row and add to NewSamples _without_ data copy
                cv::Mat Sample(1, input[0].size(), cv::DataType<float>::type, input[i].data());

                NewSamples.push_back(Sample);
            }

            torch::Tensor Output = torch::from_blob(NewSamples.data, {(long int)input.size(), (long int)input[0].size()}).contiguous().clone();

            return Output;
        }
    }

    bool aiutils::checkFileExists(std::string Filename)
    {
        return access(Filename.c_str(), 0) == 0;
    }

    //convert string to enum
    MODE aiutils::setMode(string Mode)
    {

        if (Mode == "TensorRT")
        {
            cout<<"setMode" << endl;
            return TensorRT;
        }

        if (Mode == "Cpu")
        {

            return Cpu;
        }
    }
//convert enum to string
    string aiutils::setYamlMode(MODE t)
    {

        cout<<"setYamMode "<< t <<endl;
        if (t == TensorRT)
        {

            return "TensorRT";
        }

        if (t == Cpu)
        {

            return "Cpu";
        }
    }

//This fuction check if Mode is implementend before instantiate onnxruntime Session
bool aiutils::checkMode(MODE m, string &Message){

    switch (m)
    {

    case TensorRT:
        Message="Mode selected TensorRT";
        break;
    case Cpu:
        Message="Mode selected Cpu";
        break;
    default:
        Message= "Mode NOT IMPLEMENTED cannot continue";
        break;
    }

}

} // namespace aiProductionReady