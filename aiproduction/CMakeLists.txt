cmake_minimum_required(VERSION 3.13)



message("INCLUDE VCPKG")
project(aiproduction)

include(GenerateExportHeader)

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

find_package(yaml-cpp CONFIG REQUIRED)

message("libtorch Path ${Torch_DIR}")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# windows Only

#opencv
if(${X86_64})
    if(UNIX)
    find_package(OpenCV 4.1 EXACT REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/)
        endif()
    if (MSVC)
        find_package(OpenCV 4.4 EXACT REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/)
    endif()
elseif(${AARCH64})
    find_package(OpenCV 4.1 REQUIRED)
endif()


add_library(aiproduction SHARED classification.cpp classification.h objectdetection.h objectdetection.cpp utils.h utils.cpp define.h aiproduction_export.h)

#onnx runtime


#linux
#target_link_libraries(aiproduction "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

#windows
target_link_libraries(aiproduction "${TORCH_LIBRARIES}" ${OpenCV_LIBS} onnxruntime yaml-cpp )



# crea gli export necessari per includere le librerie sia su windows che su linux 
#aggiungo questo file a gitignore siccome è generato a runtime
generate_export_header(aiproduction EXPORT_FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/aiproduction_export.h)

if (MSVC)
  #libtorch
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET aiproduction
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:aiproduction>)
    #tensorrt 
   file(GLOB TENSORRT_DLL "C:/tensorrt/lib/*.dll")
   add_custom_command(TARGET aiproduction
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TENSORRT_DLL}
                     $<TARGET_FILE_DIR:aiproduction>)

    file(GLOB ONNXRUNTIME_DLL "../onnxruntime/lib/onnxruntime/tensorrt/*.dll")
    add_custom_command(TARGET aiproduction
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${ONNXRUNTIME_DLL}
                    $<TARGET_FILE_DIR:aiproduction>)
endif (MSVC)



message("shared Library")
