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

        std::cout << img_tensor.dim() << std::endl;
        std::cout << img_tensor.sizes() << std::endl;

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

//sort index function
    template <typename T>
    std::deque<size_t> aiutils::sortIndexes(const std::vector<T> &v)
    {

        std::deque<size_t> indices(v.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::stable_sort(std::begin(indices), std::end(indices), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

        return indices;
    }

//non max suppression
    std::vector<uint64_t> aiutils::nms(const std::vector<std::array<float, 4>> &bboxes,           //
                                       const std::vector<float> &scores,                          //
                                       const float overlapThresh,                          //
                                       const uint64_t topK  //
    )
    {
        assert(bboxes.size() > 0);
        uint64_t boxesLength = bboxes.size();
        const uint64_t realK = std::max(std::min(boxesLength, topK), static_cast<uint64_t>(1));

        std::vector<uint64_t> keepIndices;
        keepIndices.reserve(realK);

        std::deque<uint64_t> sortedIndices = aiutils::sortIndexes(scores);

        // keep only topk bboxes
        for (uint64_t i = 0; i < boxesLength - realK; ++i)
        {
            sortedIndices.pop_front();
        }

        std::vector<float> areas;
        areas.reserve(boxesLength);
        std::transform(std::begin(bboxes), std::end(bboxes), std::back_inserter(areas),
                       [](const auto &elem) { return (elem[2] - elem[0]) * (elem[3] - elem[1]); });

        while (!sortedIndices.empty())
        {
            uint64_t currentIdx = sortedIndices.back();
            keepIndices.emplace_back(currentIdx);

            if (sortedIndices.size() == 1)
            {
                break;
            }

            sortedIndices.pop_back();
            std::vector<float> ious;
            ious.reserve(sortedIndices.size());

            const auto &curBbox = bboxes[currentIdx];
            const float curArea = areas[currentIdx];

            std::deque<uint64_t> newSortedIndices;

            for (const uint64_t elem : sortedIndices)
            {
                const auto &bbox = bboxes[elem];
                float tmpXmin = std::max(curBbox[0], bbox[0]);
                float tmpYmin = std::max(curBbox[1], bbox[1]);
                float tmpXmax = std::min(curBbox[2], bbox[2]);
                float tmpYmax = std::min(curBbox[3], bbox[3]);

                float tmpW = std::max<float>(tmpXmax - tmpXmin, 0.0);
                float tmpH = std::max<float>(tmpYmax - tmpYmin, 0.0);

                const float intersection = tmpW * tmpH;
                const float tmpArea = areas[elem];
                const float unionArea = tmpArea + curArea - intersection;
                const float iou = intersection / unionArea;

                if (iou <= overlapThresh)
                {
                    newSortedIndices.emplace_back(elem);
                }
            }

            sortedIndices = newSortedIndices;
        }

        return keepIndices;
    }

} // namespace aiProductionReady