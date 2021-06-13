## Ai4prod 👋

**Ai4prod is the first ecosystem built for offering an end-to-end solution to handle AI project in production environment using C++** 

  

**The core design Principles** of Ai4prod are:

  

- Easy **Integration**

- Easy **Customization**

- Works on **different Hardware** and operating systems (**Windows,Linux,Jetson**)

- Make coding workflow standard for code **maintainability**

- C++ ready


**Ai4prod is built for**

  

- Newbie Machine Learning Engineers, who feel lost on how to bring Ai project in production.

- Experienced Machine Learning Engineers, who are looking for an easy way to use their **custom model with C++ in production**.

  
  
**Who developed Ai4prod?**

Ai4Prod is maintaned by a team of Machine Learning engineers, that everyday, are trying to bring and maintain Machine Learning project in production using C++

  

**Why Ai4Prod is different**?

Ai4prod is offering you the entire pipeline from training to inference. So learning how to use Ai4Prod gives you the ability to deliver real value with AI.

Ai4prod is fully tested, so you don't need to worry about code compatibiity, accuracy between different libraries version. We handle all for you.

  
 We developed ai4prod with the idea to simplify the entire pipeline in a machine learning project. As a machine learning engineers, we know that make something works in Ai project is not simple.

We built ai4prod to help us to crete real value in a production environment, so we hope that could also help you.

Ai4prod is developed following our experience in production, if you think that something is missing or could be helpfull drop us an email ai **info@ai4prod.ai** or open a thread on github.

We are always open to collaboration.

  **Medium Article** https://ai4prod.medium.com/ai4prod-the-first-ecosystem-to-bring-ai-to-production-in-c-8abb0d2f9424  

## Getting Started

  

Ai4prod is built to be as easy as possible to get started. You can reach our website at the following link https://www.ai4prod.ai/, where you can find all the documentation. Here you can find a quick guide.


 ### Prerequisites
 1.  For all platforms you need to install cuda and cudnn, IF you want to use GPU acceleration. If you are not familiar on how to install we create a guide https://www.ai4prod.ai/docs/error/tip-tricks/install-cuda-with-cuda-toolkit/
 2. You need to install CMake >3.13 to compile ai4prod. If you don't know how to do, follow our guide https://www.ai4prod.ai/docs/getting-started/. Choose your operating system

### Install dependencies
We provide installation script for bot Windows and Linux

### STEP 1
#### Windows 

   `./installWin.bat --cuda 11.0`

####  Linux 

    ./install.sh --cuda 10.2 --cmake 
    

 - -- cmake is optional. Will install cmake 3.14 and override previous version

### STEP 2

Compile the project with Cmake

#### Linux 

    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_INSTALL_PREFIX= {your_install_folder} -DEXECUTION_PROVIDER=tensorrt|cpu -DCMAKE_BUILD_TYPE=Release ..
    $ make -j10
    $ make install 
 For a complete tutorial have a look here https://www.ai4prod.ai/docs/getting-started/build-for-linux/  
#### Windows

For a complete tutorial have a look here
https://www.ai4prod.ai/docs/getting-started/build-for-windows/
#### Jetson
For a complete tutorial have a look here

https://www.ai4prod.ai/docs/getting-started/build-for-jetson/

## Tutorial
We create different tutorials to let you start developing with Ai4Prod on your custom project. All examples are available in the example folder. Before run any example you need to download the related .onnx model. 

### Models
from this link https://drive.google.com/drive/folders/1a8m6Ek8qH7eKg8W9CdGQJ3_qeCeYx026?usp=sharing

### Prerequisite
You need to install Ai4Prod first.
  

### Classification

 https://www.ai4prod.ai/docs/tutorial/resnet50/


### Object Detection

https://www.ai4prod.ai/docs/tutorial/yolo

### Instance Segmentation
https://www.ai4prod.ai/docs/tutorial/yolact/

### Pose Estimation
https://www.ai4prod.ai/docs/tutorial/hrnet/


## Results

### Inference Time

Result are FPS on 1000 iterations

|Model |GPU/Backend  |  CPU | FP32 |FP 16| OS| 
|--|--|--|--|--|--|
| yolov3spp-608 |2070/Tensorrt  |0.9  |42,54 |125,13|Ubuntu 18.04|
|yolov4-608|2070/Tensorrt||41,66|124,96|Ubuntu 18.04|
|yolov3spp-608|2070/Tensorrt||41,23|115,23|Win 10|
|yolov3spp-608|XavierNX||5.18|18.86|Jetpack 4.4|
|yoalact-resnet50|2070/Tensorrt|2.8|37,03|102,04|ubuntu 18.04|
|resnet50|2070/Tensorrt|14|320,43|667,63|Win 10|
|resnet50-base|Xavier NX||50,96|83,74|Jetpack 4.4|


### Accuracy

| Model |Dataset |Metrics|Backend|FP32|FP16|OS|
|--|--|--|--|--|--|--|
| yolov3-spp-base-608 |Coco 2017|MAP(AP50)|Tensorrt|66.1|66.1|ubuntu 18.04|
|yolov4-608|Coco 2017|MAP(AP50)|Tensorrt|72.3|72.3|ubuntu 18.04|
|yolov3-spp-base-608|Coco 2017|MAP(AP50)|Tensorrt|66.1|66.1|Windows 10|
|yolov3-spp-base-608|Coco 2017|MAP(AP50)|Tensorrt|65.1|65.2|Jetson Xavier|
|resnet50-base|Imagenet 2012|Accuracy|Tensorrt|75%-92%|74%-92%|ubuntu 18.04|
|resnet50-base|Imagenet 2012|Accuracy|Tensorrt|75%-92%|74%-92%|Windows 10|
|yolact-resnet50-550|Coco 2017|MAP(AP50)|Tensorrt|42.1|41.9|ubuntu 18.04|
|yolact-resnet50-550|Coco 2017|MAP(AP50)|Tensorrt|42.1|41.9|Windows 10|
|yolact-resnet50-550|Coco 2017|MAP(AP50)|Tensorrt|36.1||Jetson Xavier|

## Troubleshooting

If you encounter some problems you can open an issue on github or send us an email


