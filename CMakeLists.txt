

#set(VCPKG_TARGET_TRIPLET "x64-linux")
#vcpg set before project
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")

cmake_minimum_required(VERSION 3.13)

project(aiprod)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)
#architecture definition

if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
    set(X86_64 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
    set(AARCH64 1)
endif()

if(UNIX)
set (DEFAULT_PATH 1)
endif()
#Where to install the library

string(COMPARE EQUAL "${DEFAULT_PATH}" 1 result)

# user not specified installation path

if(result)


	if(${X86_64})
    

		if(UNIX)
			#set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/example/Linux/install-Lib")
			set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/example/Linux/classification/Install")
		endif()
		if (MSVC)
			#example
			set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/example/Windows/objectdetection/Install")
			#accuracy
			#set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/testAccuracy/objectDetection/TestCoco/Cpp/Windows/Install")
		endif()

	elseif(${AARCH64})
			#accuracy testAccuracy/classification/Imagenet/Cpp/Linux
			set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/testAccuracy/objectDetection/TestCoco/Cpp/Linux/Install")
			#set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/example/Aarch64/classification/Install")
	endif()

endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/deps/opencv/lib")

# library to generate export header
include(GenerateExportHeader)

set(ONNXRUNTIME_INCLUDE_DIR deps/onnxruntime/include/)

#onnxruntime build with tensorrt 
# tensorrt library must be in /usr/local/tensorrt for linux
link_directories(deps/onnxruntime/lib/onnxruntime/tensorrt)

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
        find_package(OpenCV 4.1 EXACT REQUIRED PATHS deps/opencv/)
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

# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()


message("JSON CPP ${JSON_CPP_INCLUDE_DIRS}")


message("YAML-CPP ${YAML_CPP_INCLUDE_DIR}")
#ai4production Library
add_library(aiproduction SHARED aiproduction/classification.cpp 

aiproduction/classification.h 
aiproduction/objectdetection.h 
aiproduction/objectdetection.cpp 
aiproduction/utils.h 
aiproduction/utils.cpp 
aiproduction/aiproduction_export.h 
aiproduction/modelInterface.h)

if(UNIX)
        
        target_link_libraries(aiproduction "${TORCH_LIBRARIES}" ${OpenCV_LIBS} onnxruntime yaml-cpp jsoncpp_static stdc++fs)
    endif()
    if (MSVC)
    target_link_libraries(aiproduction "${TORCH_LIBRARIES}" ${OpenCV_LIBS} onnxruntime yaml-cpp jsoncpp )
endif()



#need for symbol visibility
generate_export_header(aiproduction EXPORT_FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/aiproduction/aiproduction_export.h)


#Library Version

set_target_properties(
  aiproduction PROPERTIES
    SOVERSION 0
    VERSION 0.4.1
)

#Installing

set(AIPRODUCTION_INCLUDES aiproduction/classification.h 
aiproduction/objectdetection.h 
aiproduction/utils.h 
aiproduction/aiproduction_export.h aiproduction/modelInterface.h
)

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


