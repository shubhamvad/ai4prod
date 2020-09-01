cmake_minimum_required(VERSION 3.13)

project(interface)

set(CMAKE_CXX_STANDARD 14)


#onnxruntime
set(ONNXRUNTIME_INCLUDE_DIR onnxruntime/include/)

message(${ONNXRUNTIME_INCLUDE_DIR})

link_directories(onnxruntime/lib/onnxruntime)

include_directories(
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${ONNXRUNTIME_INCLUDE_DIR}onnxruntime/core/session
)

# torchVision

#serve per includere torchvision
#set(TorchVision_DIR /home/tondelli/Desktop/2020/aiproductionready/torchvision/share)




#serve per il find_package di includere il progetto libtorch
set(Torch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch/)

# ritorna direttamente gli include senza bisogno di aggiungere target_include_directories
find_package(Torch REQUIRED)

#find_package(TorchVision REQUIRED)

# servono per compilare libtorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


#OpenCv

find_package(OpenCV 4.1 EXACT REQUIRED PATHS opencv/)


# aggiunge la libreria di classficazione del framework
#add_library(classification SHARED src/classification.cpp src/classification.h )

add_subdirectory(aiproduction)


add_executable(interface main.cpp)

# include torchvision
#target_link_libraries(interface aiproduction "${TORCH_LIBRARIES}" ${OpenCV_LIBS} onnxruntime custom_op_library TorchVision::TorchVision)

target_link_libraries(interface aiproduction "${TORCH_LIBRARIES}" ${OpenCV_LIBS} onnxruntime onnxruntime_providers_tensorrt onnxruntime_providers_shared)

