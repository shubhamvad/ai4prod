cmake_minimum_required(VERSION 3.12)

project (inference)

#need for LibTorch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

set (aiproduction_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Install/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(aiproduction REQUIRED)

#link_directories(inference install-Lib/lib)

add_executable(inference classification.cpp)

if(UNIX)
target_link_libraries(inference ${AIPRODUCTION_LIBS} stdc++fs)
endif()


if (MSVC)
target_link_libraries(inference ${AIPRODUCTION_LIBS} )
endif()

message("AI EXECUTION PROVIDER ${AI_EXECUTION_PROVIDER}")

target_include_directories(inference PRIVATE ${AIPRODUCTION_INCLUDE_DIRS} )

if (MSVC)
  file(GLOB TORCH_DLLS "Install/deps/libtorch/lib/*.dll")
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
     
                     
    file(GLOB ONNXRUNTIME_DLL "Install/deps/onnxruntime/${AI_EXECUTION_PROVIDER}/lib/onnxruntime/*.dll")
    add_custom_command(TARGET inference
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${ONNXRUNTIME_DLL}
                     $<TARGET_FILE_DIR:inference>)

  file(GLOB AIPROD_DLL "Install/bin/*.dll")
  add_custom_command(TARGET inference
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${AIPROD_DLL}
                     $<TARGET_FILE_DIR:inference>)
    
    
            
endif (MSVC)