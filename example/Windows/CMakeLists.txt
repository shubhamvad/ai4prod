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


if (MSVC)
  file(GLOB TORCH_DLLS "install-Lib/deps/libtorch/lib/*.dll")
  add_custom_command(TARGET inference
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:inference>)
     #tensorrt 
   file(GLOB TENSORRT_DLL "C:/tensorrt/lib/*.dll")
   add_custom_command(TARGET inference
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TENSORRT_DLL}
                     $<TARGET_FILE_DIR:inference>)
     
                     
    file(GLOB ONNXRUNTIME_DLL "install-Lib/deps/onnxruntime/lib/onnxruntime/tensorrt/*.dll")
    add_custom_command(TARGET inference
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${ONNXRUNTIME_DLL}
                     $<TARGET_FILE_DIR:inference>)

  file(GLOB AIPROD_DLL "install-Lib/bin/*.dll")
  add_custom_command(TARGET inference
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${AIPROD_DLL}
                     $<TARGET_FILE_DIR:inference>)
    
    
            
endif (MSVC)