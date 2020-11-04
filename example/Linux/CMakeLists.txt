cmake_minimum_required(VERSION 3.12)

project (inference)

#serve per compilare LibTorch
set(CMAKE_CXX_STANDARD 14)

set (aiproduction_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install-Lib/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(aiproduction REQUIRED)

#link_directories(inference install-Lib/lib)

add_executable(inference main.cpp)

target_link_libraries(inference ${AIPRODUCTION_LIBS} )

target_include_directories(inference PRIVATE ${AIPRODUCTION_INCLUDE_DIRS} )