# Prerequisite

- Install ai4prod library in this folder under Install directory. If you don't know how, have a look 
at this tutorial https://www.ai4prod.ai/docs/getting-started/build-for-jetson/

- create tensorrtModel folder (without this folder Tensorrt model is not saved)


# How to compile

Open a terminal and execute from this folder

	$ mkdir build
	$ cd build
	$ cmake ..
	$ make -j8


# How to run

Open a terminal and inside build folder run ./inference



