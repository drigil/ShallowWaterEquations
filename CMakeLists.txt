cmake_minimum_required(VERSION 3.18)

PROJECT(SWE LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(SOURCES 
		src/main.cu
		src/SWE.cu
		src/utils.cpp
		src/utils.h
		)

set(GLM_DIR    ${CMAKE_SOURCE_DIR}/depends/glm)
set(IMGUI_ROOT ${CMAKE_SOURCE_DIR}/depends/imgui)
set(GL3W_ROOT  ${CMAKE_SOURCE_DIR}/depends/gl3w)

set(IMGUI_SOURCES ${IMGUI_ROOT}/imgui.cpp ${IMGUI_ROOT}/imgui_draw.cpp ${IMGUI_ROOT}/imgui_widgets.cpp)
set(IMGUI_IMPL_SOURCES ${IMGUI_ROOT}/imgui_impl_glfw.cpp ${IMGUI_ROOT}/imgui_impl_opengl3.cpp)
set(GL3W_SOURCES ${GL3W_ROOT}/GL/gl3w.c)

add_executable(${PROJECT_NAME} ${SOURCES} ${IMGUI_SOURCES} ${IMGUI_IMPL_SOURCES} ${GL3W_SOURCES})
add_definitions(-lcuda)
add_definitions(-lcudart)
target_link_libraries(${PROJECT_NAME} CUDA::cudart)

add_subdirectory(depends/glfw)
target_link_libraries(${PROJECT_NAME} glfw)

find_package(OpenGL REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${OPENGL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES})

target_include_directories(${PROJECT_NAME} PUBLIC ${IMGUI_ROOT})
target_include_directories(${PROJECT_NAME} PUBLIC ${GLM_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${GL3W_ROOT})
