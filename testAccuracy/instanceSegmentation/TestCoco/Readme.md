#Coco 2017 Map detection

Coco2017.py is to used to test the Map beetween instances_val2017.json and the output from our object detection alogirithms in c++

# How to get started

- Download Coco validation dataset 2017 
- Copy the file instances_val2017.json if not present in this repository
- Compile your repository and install in this folder or once compiled copy the install file in this folder under folder Install/. Create folder Install if not present

# How to run object detection in Cpp

- Open the file testCoco.cpp and set al the required Path
- Activate evalutation mode id commented
- create a build folder
- cd build
- cmake ..
- make
- execute ./testCoco
- At the end the script create yoloVal.json

# How to get Map result

- verify that you have instances_val2017.json and yoloVal.json in this folder
- from shell run Coco2017.py 
- if you get error on some packages just install using pip


# Execution Error

/home/aistudios/Develop/ai4prod/test/objectDetection/TestCoco/build/testcocomap: error while loading shared libraries: libopencv_imgproc.so.4.1: cannot open shared object file: No such file or directory

Set to .bashrc

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aistudios/Develop/ai4prod/test/objectDetection/TestCoco/Install/deps/opencv/lib


# Note

For object detection the file is saved only if EVAL_ACCURACY is defined and createAccuracyFile() is called at the end

The detection accuracy need to be > 0.01

NMS should be set to 0.5
