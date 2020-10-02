cmake_minimum_required(VERSION 3.13)

project(interface)

set(CMAKE_CXX_STANDARD 14)

#verifica archiettura
if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
    set(X86_64 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
    set(AARCH64 1)
endif()


#onnxruntime
set(ONNXRUNTIME_INCLUDE_DIR onnxruntime/include/)

message(${ONNXRUNTIME_INCLUDE_DIR})

link_directories(onnxruntime/lib/onnxruntime/tensorrt)

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

# windows Only



#OpenCv

#OpenCv
if(${X86_64})
    
    if(UNIX)
        find_package(OpenCV 4.1 EXACT REQUIRED PATHS opencv/)
    endif()
    if (MSVC)
        find_package(OpenCV 4.4 EXACT REQUIRED PATHS opencv/)
    endif()
elseif(${AARCH64})
    find_package(OpenCV 4.1 REQUIRED)
endif()

# aggiunge la libreria di classficazione del framework
#add_library(classification SHARED src/classification.cpp src/classification.h )

add_subdirectory(aiproduction)


add_executable(interface main.cpp)

# include torchvision
#target_link_libraries(interface aiproduction "${TORCH_LIBRARIES}" ${OpenCV_LIBS} onnxruntime custom_op_library TorchVision::TorchVision)

#target_link_libraries(interface aiproduction "${TORCH_LIBRARIES}" ${OpenCV_LIBS} onnxruntime onnxruntime_providers_tensorrt onnxruntime_providers_shared)


#linux

#target_link_libraries(interface aiproduction "${TORCH_LIBRARIES}" ${OpenCV_LIBS}  onnxruntime )

#windows 
target_link_libraries(interface aiproduction "${TORCH_LIBRARIES}" ${OpenCV_LIBS} onnxruntime )
#target_link_libraries(interface "${TORCH_LIBRARIES}" ${OpenCV_LIBS} onnxruntime)


if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET interface
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:interface>)
     #tensorrt 
   file(GLOB TENSORRT_DLL "C:/tensorrt/lib/*.dll")
   add_custom_command(TARGET interface
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TENSORRT_DLL}
                     $<TARGET_FILE_DIR:interface>)
     
                     
    file(GLOB ONNXRUNTIME_DLL "onnxruntime/lib/onnxruntime/tensorrt/*.dll")
    add_custom_command(TARGET interface
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${ONNXRUNTIME_DLL}
                     $<TARGET_FILE_DIR:interface>)
    
    
            
endif (MSVC)