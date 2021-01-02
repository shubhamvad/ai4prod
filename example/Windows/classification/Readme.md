# Prerequisite

Install ai4prod library in this folder under Install directory. If you don't know how have a look 
at this tutorial https://www.ai4prod.ai/build-ai4prod-inference-for-windows/

#How to compile

Open a terminal and execute from this folder

	$ mkdir build
	$ cd build
	$ cmake -G "Visual Studio 15 2017" -A x64 ..  (you can choose also Visual Studio 16 2019 )
	$ cmake --build . --config Release


#How to run

Before run the program you need to copy from {Ai4prod_HOME}/build/vcpkg_installed/x64-windows/bin/*.dll all dll file


Note: Inside build folder you will find inference.sln project
