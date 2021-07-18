# Introduction

Here you can find example on how to run classification model in C++

- ImageNet folder = Example on how to use standard Imagenet pipeline
- CustomModel folder = Example on how to customize Ai4Prod if you have trained a custom model 

# Prerequisite

- Install ai4prod library in {ImageNet or CustomModel}  under Install directory. If you don't know how, have a look 
at this tutorial https://www.ai4prod.ai/docs/getting-started/

- If you select tensorrt Mode the optimized model is saved in your specified directory

#Model 

https://drive.google.com/file/d/1y1Pcz_u0N1GYY-cAga4E0-Qpm6nTpA9T/view?usp=sharing



# How to compile

Open a terminal and execute from CustomModel or ImageNet folder

	$ mkdir build
	$ cd build
	$ cmake ..
	$ make -j8


# How to run

Open a terminal and inside build folder run ./inference





