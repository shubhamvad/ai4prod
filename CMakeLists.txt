

#set(VCPKG_TARGET_TRIPLET "x64-linux")
#vcpg set before project
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")

cmake_minimum_required(VERSION 3.10)

project(aiprod)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)

# check for execution provider 
# default cpu
if( NOT DEFINED EXECUTION_PROVIDER)

    set (EXECUTION_PROVIDER "cpu")

endif()
#architecture definition

# this set #define TENSORRT from Cmake



if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
    set(X86_64 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
    set(AARCH64 1)
endif()

#extra include directories
set(AIPRODUCTION_INCLUDE_EXTRA "")

#Where to install the library

string(COMPARE EQUAL "${TEST_ACCURACY}" 1 accuracy)

# user not specified installation path


if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/Install )
 
endif()


# set Rpath for runtime
set(MY_INSTALL_RPATH
"${CMAKE_INSTALL_PREFIX}/deps/opencv/lib"
"${CMAKE_INSTALL_PREFIX}/deps/onnxruntime/${EXECUTION_PROVIDER}/lib/onnxruntime"
"${CMAKE_INSTALL_PREFIX}/deps/libtorch/lib"
)


set(CMAKE_INSTALL_RPATH "${MY_INSTALL_RPATH}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# library to generate export header
include(GenerateExportHeader)

set(ONNXRUNTIME_INCLUDE_DIR deps/onnxruntime/${EXECUTION_PROVIDER}/include/)

#onnxruntime build with tensorrt 
# tensorrt library must be in /usr/local/tensorrt for linux
link_directories(deps/onnxruntime/${EXECUTION_PROVIDER}/lib/onnxruntime/)

if(MSVC)
# this add folder path to .lib library in linker directory
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/build/vcpkg_installed/x64-windows/lib)
endif()

include_directories(
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${ONNXRUNTIME_INCLUDE_DIR}onnxruntime/core/session
)

#Libtorch
set(Torch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/libtorch/share/cmake/Torch/)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#opencv usually without contrib

if(${X86_64})
    
    if(UNIX)
        find_package(OpenCV 4.4 EXACT REQUIRED PATHS deps/opencv/)
    endif()
    if (MSVC)
        find_package(OpenCV 4.4 EXACT REQUIRED PATHS deps/opencv/)
    endif()
elseif(${AARCH64})
    find_package(OpenCV 4.1 REQUIRED)
endif()


#yaml-cpp
find_package(yaml-cpp CONFIG REQUIRED) 

#json-cpp

find_package(jsoncpp CONFIG REQUIRED)

#catch2

find_package(Catch2 CONFIG REQUIRED)
    

# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()


message("JSON CPP ${JSON_CPP_INCLUDE_DIRS}")


message("YAML-CPP ${YAML_CPP_INCLUDE_DIR}")
#FILE(GLOB MY_SRCS dir/*)

FILE(GLOB MY_SRCS_INCLUDE 
aiproduction/include/classification.h 
aiproduction/include/objectdetection.h 
aiproduction/include/instancesegmentation.h 
aiproduction/include/utils.h 
aiproduction/include/aiproduction_export.h 
aiproduction/include/modelInterface.h
aiproduction/include/customDataType.h
aiproduction/include/posedetection.h
aiproduction/include/semanticsegmentation.h
aiproduction/include/defines.h
)

FILE(GLOB MY_SRCS 
aiproduction/src/classification.cpp 
aiproduction/src/objectdetection.cpp
aiproduction/src/instancesegmentation.cpp
aiproduction/src/posedetection.cpp 
aiproduction/src/utils.cpp
aiproduction/src/semanticsegmentation.cpp
)

#ai4production Library
add_library(aiproduction SHARED ${MY_SRCS} ${MY_SRCS_INCLUDE})

# set_target_properties(aiproduction
#     PROPERTIES
#     INSTALL_RPATH "${MY_INSTALL_RPATH}"
# )
target_include_directories(aiproduction PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/aiproduction/include )


if(UNIX)        
    target_link_libraries(aiproduction "${TORCH_LIBRARIES}" ${OpenCV_LIBS} onnxruntime yaml-cpp jsoncpp_static stdc++fs)
endif()

if (MSVC)
    target_link_libraries(aiproduction "${TORCH_LIBRARIES}" ${OpenCV_LIBS} onnxruntime yaml-cpp jsoncpp )
endif()

# set Cmake preprocessor definition
if(accuracy)
    message("WARNING. LIBRARY BUILD FOR ACCURACY EVALUATION. DO NOT USE IN PRODUCTION")
    #This Flag enable accuracy evaluation
    target_compile_definitions(aiproduction PRIVATE EVAL_ACCURACY)
	#set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/testAccuracy/classifcation/TestCoco/Cpp/Install")
		
endif()


#create executable for test
if(ADDTEST STREQUAL "ON")
    message("ADD TEST")
    find_package(CUDA  REQUIRED)
    add_executable(catch_test test/testClassification.cpp test/testObjectDetection.cpp test/testInstanceSegmentation.cpp test/testPoseDetection.cpp  ${MY_SRCS_INCLUDE} )
   

    target_include_directories(catch_test PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/aiproduction/include "${CUDA_INCLUDE_DIRS}")


    if(UNIX)
        #cuda libraries are required to release Gpu memory allocation for TensorRT session
        target_link_libraries(catch_test "${TORCH_LIBRARIES}" "${CUDA_LIBRARIES}" ${OpenCV_LIBS} aiproduction onnxruntime yaml-cpp jsoncpp_static stdc++fs Catch2::Catch2)
    endif()

    if (MSVC)
        target_link_libraries(catch_test "${TORCH_LIBRARIES}" ${OpenCV_LIBS} aiproduction onnxruntime yaml-cpp jsoncpp Catch2::Catch2 )

        file(GLOB TORCH_DLLS "deps/libtorch/lib/*.dll")
        add_custom_command(TARGET catch_test
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:catch_test>)
        #tensorrt 
    if(EXECUTION_PROVIDER STREQUAL "tensorrt")
        file(GLOB TENSORRT_DLL "C:/tensorrt/lib/*.dll")
        add_custom_command(TARGET catch_test
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TENSORRT_DLL}
                        $<TARGET_FILE_DIR:catch_test>)
    endif()    
                        
        file(GLOB ONNXRUNTIME_DLL "deps/onnxruntime/${EXECUTION_PROVIDER}/lib/onnxruntime/*.dll")
        add_custom_command(TARGET catch_test
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${ONNXRUNTIME_DLL}
                        $<TARGET_FILE_DIR:catch_test>)
    endif()

endif()

# set execution provider preprocess definition 

if(EXECUTION_PROVIDER STREQUAL "tensorrt")
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime/${EXECUTION_PROVIDER}/include)
        message("TENSORRT EXECUTION PROVIDER")
        target_compile_definitions(aiproduction PRIVATE TENSORRT)
        if(ADDTEST STREQUAL "ON")
            target_compile_definitions(catch_test PRIVATE TENSORRT)
        endif()
    else()
        message(SEND_ERROR "TENSORRT PROVIDER NOT FOUND. PLEASE DOWNLOAD IT")
    endif()
elseif(EXECUTION_PROVIDER STREQUAL "directml")
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime/${EXECUTION_PROVIDER}/include)
        message("DIRECTML EXECUTION PROVIDER")
        if(MSVC)
            target_compile_definitions(aiproduction PRIVATE DIRECTML)
            if(ADDTEST STREQUAL "ON")
                target_compile_definitions(catch_test PRIVATE DIRECTML)
            endif()
            
        else()
            message(SEND_ERROR "DIRECT ML PROVIDER ONLY WORKS UNDER WINDOWS")
        endif()
    else()
        message(SEND_ERROR "${EXECUTION_PROVIDER} PROVIDER NOT FOUND. PLEASE DOWNLOAD IT")
    endif()
elseif(EXECUTION_PROVIDER STREQUAL "cpu")
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime/${EXECUTION_PROVIDER}/include)
        message("CPU EXECUTION PROVIDER")
        target_compile_definitions(aiproduction PRIVATE AICPU)
        if(ADDTEST STREQUAL "ON")
            target_compile_definitions(catch_test PRIVATE AICPU)
        endif()
    else()
        message(SEND_ERROR "CPU PROVIDER NOT FOUND. PLEASE DOWNLOAD IT")
    endif()

else()
    message(SEND_ERROR  "NO EXECUTION PROVIDER SET. USE -DEXECUTION_PROVIDER= cpu | tensorrt | directml")
endif()

#TEST




#VERIFY EXTRA DEPENDENCIES
#Tracking
IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/extra/Tracking)

set(LIBS ${LIBS} tracking)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extra/Tracking)

endif()

#Explain
IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/extra/Explain)

set(LIBS ${LIBS} explain)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extra/Explain)

endif()

#Ocr
IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/extra/Ocr_private)

    file(GLOB RESULT ${CMAKE_CURRENT_SOURCE_DIR}/extra/Ocr_private/src/*.cpp)

    list(LENGTH RESULT RES_LEN)

    # check if directory has some file
    if(RES_LEN EQUAL 0)


    else()
        message("FOUND OCR")
        set(LIBS ${LIBS} ocr)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extra/Ocr_private)

    endif()

endif()

set(LIBS ${LIBS} aiproduction)

#need for symbol visibility
generate_export_header(aiproduction EXPORT_FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/aiproduction/include/aiproduction_export.h)


#Library Version

set_target_properties(
  aiproduction PROPERTIES
    SOVERSION 0
    VERSION 0.7.0
)

#Installing

set(AIPRODUCTION_INCLUDES ${MY_SRCS_INCLUDE})

set_target_properties(aiproduction PROPERTIES PUBLIC_HEADER "${AIPRODUCTION_INCLUDES}" )

# generate aiproductionConfig.cmake used by find_package
configure_file(aiproductionConfig.cmake.in "${CMAKE_INSTALL_PREFIX}/aiproductionConfig.cmake" @ONLY)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/deps/ DESTINATION ${CMAKE_INSTALL_PREFIX}/deps/ )

#vcpkg yaml-cpp install include
install(DIRECTORY ${YAML_CPP_INCLUDE_DIR} DESTINATION ${CMAKE_INSTALL_PREFIX})

#install ai4prod deps
install(TARGETS aiproduction
    #EXPORT aiproductionTargets.cmake
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)


#export ( TARGETS aiproduction FILE ${CMAKE_INSTALL_PREFIX}/aiproductionTargets.cmake  )


