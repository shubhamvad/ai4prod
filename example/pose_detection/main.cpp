#include <iostream>

#include "torch/torch.h"

#include "classification.h"
#include "objectdetection.h"
#include "posedetection.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace ai4prod;
using namespace objectDetection;
using namespace poseDetection;
using namespace classification;

using namespace std::chrono;

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace std;
using namespace cv;

int main()
{

    //object detection

    Yolov4 *yolov4;

    yolov4 = new Yolov4();

    //this retrive only class person from detection result tensor
    vector<int> includeClass = {0};

    //Pose detection
    Hrnet *hrnet;

    hrnet = new Hrnet();

    if (!hrnet->init("C:/Users/erict/Desktop/Official/ai4prod/example/pose_detection/hrnet.onnx", 256, 192, 80, TensorRT, "C:/Users/erict/Desktop/Official/ai4prod/example/pose_detection/tensorrtModel")) {
          
        return 0;
    }
    cv::Mat image = cv::imread("C:/Users/erict/Desktop/Official/ai4prod/example/pose_detection/image/sinner.jpg");

    if (!yolov4->init("C:/Users/erict/Desktop/Official/ai4prod/example/pose_detection/yolov4_608.onnx", 608, 608, 80, TensorRT, "C:/Users/erict/Desktop/Official/ai4prod/example/pose_detection/tensorrtModel_yolov4", &includeClass))
    {

        return 0;
    }

    yolov4->preprocessing(image);

    yolov4->runmodel();

    torch::Tensor result = yolov4->postprocessing();

    for (int i = 0; i < result.sizes()[0]; i++)
    {
        int x = result[i][0].item<int>();
        int y = result[i][1].item<int>();
        int width = result[i][2].item<int>();
        int height = result[i][3].item<int>();

        cv::Rect brect = cv::Rect(x, y, width, height);

        hrnet->preprocessing(image, brect);

        hrnet->runmodel();

        torch::Tensor poseResult = hrnet->postprocessing();

        for (int j = 0; j < poseResult.sizes()[0]; j++)
        {
            for (int i = 0; i < 17; i++)
            {

                cv::Point2f joint = cv::Point2f(poseResult[j][i][0].item<float>(), poseResult[j][i][1].item<float>());

                cv::circle(image, joint, 5, (255, 255, 255), 1);
            }
        }

        //cv::rectangle(image, brect, cv::Scalar(255, 0, 0));

        cout << "class " << result[i][5] << endl;
    }

    cv::imshow("bbox", image);
    cv::waitKey(0);
    //filter bbox with person id

    cout << "run model" << endl;

    //return (b,17,2) b= corrspond a number of people found in image

    cout << "program end" << endl;
    return 0;
}