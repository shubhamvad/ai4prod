cmake_minimum_required(VERSION 3.12)

project (explain)

#needs for libtorch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED)


set(AIPRODUCTION_INCLUDE_EXTRA_EXPLAIN
${CMAKE_CURRENT_SOURCE_DIR}/include/explain.h
${CMAKE_CURRENT_SOURCE_DIR}/src/explain.cpp
) 


add_library(explain SHARED ${AIPRODUCTION_INCLUDE_EXTRA_EXPLAIN})

target_link_libraries(explain ${OpenCV_LIBS} ${TORCH_LIBRARIES}  )

target_include_directories(explain PRIVATE ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include src)

generate_export_header(explain EXPORT_FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/include/explain_export.h)
set_target_properties(
    explain PROPERTIES
    SOVERSION 0
    VERSION 0.1.0
)


set (EXPORT_HEADER_EXPLAIN ${CMAKE_CURRENT_SOURCE_DIR}/include/explain_export.h)

set(AIPRODUCTION_INCLUDE_EXTRA_EXPLAIN ${AIPRODUCTION_INCLUDE_EXTRA_EXPLAIN} ${EXPORT_HEADER_EXPLAIN})

#install 
set_target_properties(explain PROPERTIES PUBLIC_HEADER "${AIPRODUCTION_INCLUDE_EXTRA_EXPLAIN}" )

install(TARGETS explain
    #EXPORT aiproductionTargets.cmake
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)