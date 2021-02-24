cmake_minimum_required(VERSION 3.12)

project (inference)

#need for libtorch LibTorch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

set (aiproduction_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Install/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(aiproduction REQUIRED)

#link_directories(inference install-Lib/lib)

add_executable(inference instance_segmentation.cpp)

target_link_libraries(inference ${AIPRODUCTION_LIBS} stdc++fs)


target_include_directories(inference PRIVATE ${AIPRODUCTION_INCLUDE_DIRS} )
