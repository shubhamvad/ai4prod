project(onnxtestapp)

cmake_minimum_required(VERSION 3.12)
set (CMAKE_CXX_STANDARD 11)

message( ${CMAKE_CURRENT_SOURCE_DIR} )

set(ONNXRUNTIME_INCLUDE_DIR include/)

include_directories(
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${ONNXRUNTIME_INCLUDE_DIR}onnxruntime/core/session
)
# https://stackoverflow.com/questions/8774593/cmake-link-to-external-library
LINK_DIRECTORIES(lib)

#find_library(FOO_LIB onnxruntime)

#add_library( onnxruntime SHARED IMPORTED )
#add_library(custom_op_library SHARED IMPORTED)



# include_directories(
#   ${ONNXRUNTIME_INCLUDE_DIR}
#   ${REPO_ROOT}/include/onnxruntime/core/session
# )

add_executable(onnxtestapp main.cpp)


target_link_libraries(onnxtestapp onnxruntime custom_op_library)


