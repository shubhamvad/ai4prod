# Getting Started

These examples are tested on each operating system(Win,Linux,jetson) you need only compile ai4prod for your specific platform. Have a look at Prerequisite section below.


# Prerequisite

- Install ai4prod library in this folder under Install directory. If you don't know how, have a look 
at this tutorial https://www.ai4prod.ai/docs/getting-started/. Build for your operating system


# How to compile under Linux

Open a terminal and execute from one of the example folder(classifcation,object detection ecc...)

	$ mkdir build
	$ cd build
	$ cmake ..
	$ make -j8

# How to run on Linux or Jeston

Open a terminal and inside build folder run ./inference



# How to compile under Windows

Open a terminal and execute from one of the example folder(classifcation,object detection ecc...)

	$ mkdir build
	$ cd build
	$ cmake -G "Visual Studio 15 2017" -A x64 ..  (you can choose also Visual Studio 16 2019 )
	$ cmake --build . --config Release


# How to run under Windows

Before run the program you need to copy from {Ai4prod_HOME}/build/vcpkg_installed/x64-windows/bin/*.dll all dll file to your inference.exe executable folder

To create the executable inside build folder, you will find inference.sln open it and compile as a normal visual studio project

# Issue

If you find some problems open an issue on github

