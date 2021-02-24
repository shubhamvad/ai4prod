# Prerequisite

- Install ai4prod library in this folder under Install directory. If you don't know how, have a look 
at this tutorial https://www.ai4prod.ai/build-ai4prod-inference-for-windows/

- If you select tensorrt Mode the optimized model is saved in your specified directory


# How to compile

Open a terminal and execute from this folder

	$ mkdir build
	$ cd build
	$ cmake ..
	$ make -j8


# How to run

Open a terminal and inside build folder run ./inference


