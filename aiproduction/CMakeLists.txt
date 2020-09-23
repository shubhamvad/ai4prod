cmake_minimum_required(VERSION 3.13)

project(aiproduction)


set(CMAKE_CXX_STANDARD 14)

#serve per il find_package di includere il progetto libtorch
set(Torch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libtorch/share/cmake/Torch/)

#onnxruntime
# set(ONNXRUNTIME_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime/include/)

# message(${ONNXRUNTIME_INCLUDE_DIR})

# link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime/lib/onnxruntime)

# include_directories(
#   ${ONNXRUNTIME_INCLUDE_DIR}
#   ${ONNXRUNTIME_INCLUDE_DIR}onnxruntime/core/session
# )

message("libtorch Path ${Torch_DIR}")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


#opencv
if(${X86_64})
    find_package(OpenCV 4.1 EXACT REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/)
elseif(${AARCH64})
    find_package(OpenCV 4.1 REQUIRED)
endif()


add_library(aiproduction SHARED classification.cpp classification.h objectdetection.h objectdetection.cpp utils.h utils.cpp define.h)

#onnx runtime




target_link_libraries(aiproduction "${TORCH_LIBRARIES}" ${OpenCV_LIBS} )



message("shared Library")
