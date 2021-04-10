cmake_minimum_required(VERSION 3.17)

project(GPU_Project LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

set(SOURCES 
		src/main.cu
		src/SWE.cu
		)

add_executable(${PROJECT_NAME} ${SOURCES})
add_definitions(-lcuda)
add_definitions(-lcudart)
target_link_libraries(${PROJECT_NAME} CUDA::cudart)