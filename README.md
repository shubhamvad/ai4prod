## Ai4prod 👋


**Ai4prod is the first ecosystem to make inferece in C++ simple**. 

Ai4prod lets you train your model in python and convert it to C++ while mantaining the same accuracy, but speed up the performance.

Ai4prod is working independently on **Windows, Linux, Nvidia Jetson**.

We developed ai4prod with the idea to simplify the entire pipeline in a machine learning project. As a machine learning engineer we know that make something work in an Ai project is not simple. 

We built ai4prod to help us to crete real value in a production environment, so we hope that could also help you.

Ai4prod is developed following our experience in production if you think that something is missing or could be helpfull drop us an email ai **info@ai4prod.ai**.

We are always open to collaboration.

## Getting Started

Ai4prod is built to be as easy as possible to get started.

1) Download this repository
2) Download dependencies of the project. You can find all dependencies here 
https://drive.google.com/drive/u/2/folders/1B4lXyGM2IQmj6IHFThwlJRqnMGQoTjNq
3) Build https://www.ai4prod.ai/build-ai4prod-inference-library/
4) See our example 

Note: some dependencies are not build yet. If you need send us an email to info@ai4prod.ai

To make things even easier you can find all information at our site www.ai4prod.ai

## Tutorial

### Classification



### Object Detection

https://www.ai4prod.ai/object-detection-tutorial-yolov3/


## Structure of this repository

If some directory is missed is because is not mandatory, but reflect our development environment

Below you will find folder descriptio

### ai4production/
    In this folder you will find all files of the ai4prod inference library

### build
    When you compile ai4prod usually we create a build folder

### classes

### Dataset

    Folder where we download test Dataset to check accuracy for our models

### deps
    
    Folder where we install dependencies of ai4prod library. You can downalod dependecies from here https://drive.google.com/drive/u/2/folders/1B4lXyGM2IQmj6IHFThwlJRqnMGQoTjNq

### example

    In this folder you will find example on how to use ai4prod inference library from an external application. We provide very easy example to include ai4prod library in your project as soon as possible.

### Model

    Folder where we saved converted .onnx models

### onnxConversion

    In this folder you will find all python code to convert from our training tested repository in Python to get started with ai4prod inference library. If you want to know more have a look here
    https://www.ai4prod.ai/ai4prod-ecosystem-overview/

### testAccuracy

    You will find code to reproduce our results for test accuracy. You can also use this code to verify your test accuracy. Based on your dataset.

### vcpkg

    Ai4prod inference library uses vcpkg as a package dependency manager. You need to clone vcpkg project in this folder. Have a look here https://www.ai4prod.ai/build-ai4prod-inference-library/


### CMakeLists.txt

    Is the main CMakeLists.txt in order to compile ai4prod inference library


## Troubleshooting

If you encounter some problems you can open an issue on github or send us an email



## License

For use it in commercial solution please contatct us at info@ai4prod.ai

